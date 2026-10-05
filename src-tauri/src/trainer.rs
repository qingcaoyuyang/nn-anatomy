use crate::nn::network::{ForwardOut, Network};
use crate::nn::optimizer::{adam_step, sgd_step, AdamState};
use serde::Serialize;

/// One labeled training sample after preprocessing (x = 256 pixels).
#[derive(Clone)]
pub struct LabeledSample {
    pub x: Vec<f64>,
    pub y: usize,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct EpochReport {
    pub epoch: usize,
    pub loss: f64,
    pub acc: f64,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct EvalReport {
    pub acc: f64,
    pub per_class: Vec<f64>,
    pub confusion: Vec<Vec<usize>>,
}

pub fn cross_entropy_loss(net: &Network, x: &[f64], y: usize) -> f64 {
    let f = net.forward(x);
    -f.p[y].ln()
}

/// One epoch: shuffle with deterministic LCG, one SGD/Adam step per sample.
/// Returns the mean loss and the training accuracy of this epoch.
pub fn train_epoch(
    net: &mut Network,
    state: &mut AdamState,
    samples: &[LabeledSample],
    lr: f64,
    epoch: usize,
    use_sgd: bool,
) -> EpochReport {
    let n = samples.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut s = (epoch as u64 * 2654435761 + 42) & 0x7fffffff;
    for i in (1..n).rev() {
        s = s.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
        let j = ((s as f64 / 0x7fffffff as f64) * (i as f64 + 1.0)) as usize % (i + 1);
        order.swap(i, j);
    }
    let mut loss_sum = 0.0;
    let mut correct = 0;
    for idx in order {
        let s = &samples[idx];
        let f = net.forward(&s.x);
        let argmax = argmax(&f.p);
        if argmax == s.y { correct += 1; }
        loss_sum += -f.p[s.y].ln();
        let g = net.backward(&s.x, s.y, &f);
        if use_sgd {
            sgd_step(net, &g, lr);
        } else {
            adam_step(net, state, &g, lr);
        }
    }
    EpochReport { epoch, loss: loss_sum / n as f64, acc: correct as f64 / n as f64 }
}

pub fn evaluate(net: &Network, samples: &[LabeledSample]) -> EvalReport {
    let outputs = net.arch.outputs;
    let mut confusion = vec![vec![0usize; outputs]; outputs];
    let mut per_class = vec![(0usize, 0usize); outputs];
    for s in samples {
        let f = net.forward(&s.x);
        let pred = argmax(&f.p);
        confusion[s.y][pred] += 1;
        per_class[s.y].1 += 1;
        if pred == s.y {
            per_class[s.y].0 += 1;
        }
    }
    let total: usize = per_class.iter().map(|(_, t)| t).sum();
    let correct: usize = confusion
        .iter()
        .enumerate()
        .map(|(y, row)| row[y])
        .sum();
    EvalReport {
        acc: correct as f64 / total.max(1) as f64,
        per_class: per_class
            .iter()
            .map(|(c, t)| if *t > 0 { *c as f64 / *t as f64 } else { 0.0 })
            .collect(),
        confusion,
    }
}

pub fn forward_full(net: &Network, x: &[f64]) -> ForwardOut {
    net.forward(x)
}

fn argmax(p: &[f64]) -> usize {
    (0..p.len()).max_by(|a, b| p[*a].partial_cmp(&p[*b]).unwrap()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::network::{Arch, Network};
    use crate::sample_data;

    fn param_count(net: &Network) -> usize {
        let Arch { inputs, hidden, outputs } = net.arch;
        hidden * inputs + hidden + outputs * hidden + outputs
    }

    fn tiny_dataset(n_per_class: usize) -> Vec<LabeledSample> {
        let mut out = Vec::new();
        for y in 0..10 {
            for i in 0..n_per_class {
                let mut x = vec![0.0; 256];
                // Sparse synthetic pattern: two pixels encode class + variation.
                x[y * 16 + 3] = 1.0;
                x[160 + i / 4] = 0.8;
                out.push(LabeledSample { x, y });
            }
        }
        out
    }

    #[test]
    fn bundled_dataset_reaches_90_percent_via_train_epoch() {
        let ds = sample_data::load().unwrap();
        let up = |v: &Vec<f64>| crate::commands::upsample13(v);
        let samples: Vec<LabeledSample> = ds.train
            .iter()
            .map(|s| LabeledSample { x: up(&s.pixels), y: s.label as usize })
            .collect();
        let test: Vec<LabeledSample> = ds.test
            .iter()
            .map(|s| LabeledSample { x: up(&s.pixels), y: s.label as usize })
            .collect();
        let mut net = Network::new(Arch { inputs: 256, hidden: 24, outputs: 10 }, 999);
        let mut state = AdamState::new(param_count(&net));
        for epoch in 0..150 {
            train_epoch(&mut net, &mut state, &samples, 0.001, epoch, false);
        }
        let ev = evaluate(&net, &test);
        assert!(ev.acc > 0.90, "accuracy {:.4} below 90%", ev.acc);
    }

    #[test]
    fn loss_decreases_over_epochs() {
        let mut net = Network::new(Arch { inputs: 256, hidden: 24, outputs: 10 }, 7);
        let mut state = AdamState::new(param_count(&net));
        let samples = tiny_dataset(10);
        let first = train_epoch(&mut net, &mut state, &samples, 0.01, 0, false);
        let last = train_epoch(&mut net, &mut state, &samples, 0.01, 9, false);
        assert!(last.loss < first.loss, "loss must decrease: {} -> {}", first.loss, last.loss);
    }

    #[test]
    fn confusion_diagonal_matches_per_class() {
        let net = Network::new(Arch { inputs: 256, hidden: 24, outputs: 10 }, 7);
        let samples = tiny_dataset(5);
        let ev = evaluate(&net, &samples);
        assert_eq!(ev.confusion.len(), 10);
        assert_eq!(ev.confusion[0].len(), 10);
        for (y, row) in ev.confusion.iter().enumerate() {
            let sum: usize = row.iter().sum();
            assert_eq!(sum, 5);
        }
        let diag: usize = ev.confusion.iter().enumerate().map(|(y, r)| r[y]).sum();
        let correct_acc_sum: f64 = ev.per_class.iter().sum();
        assert!((diag as f64 / 50.0) - (correct_acc_sum / 10.0) < 1e-12);
    }

    #[test]
    fn sgd_and_adam_both_train() {
        let samples = tiny_dataset(10);
        let mut net_a = Network::new(Arch { inputs: 256, hidden: 24, outputs: 10 }, 7);
        let mut net_s = net_a.clone();
        let mut state = AdamState::new(param_count(&net_a));
        for e in 0..5 {
            train_epoch(&mut net_a, &mut state, &samples, 0.001, e, false);
            train_epoch(&mut net_s, &mut state, &samples, 0.1, e, true);
        }
        assert!(evaluate(&net_a, &samples).acc > 0.5);
        assert!(evaluate(&net_s, &samples).acc > 0.5);
    }
}
