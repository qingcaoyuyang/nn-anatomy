#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// Commands live in the library; re-export them so the generate_handler!
// macro can see the per-command helper macros it generates.
pub use nn_anatomy::commands::*;

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            app_init,
            dataset_add,
            dataset_clear_all,
            dataset_import_builtin,
            dataset_list,
            dataset_remove,
            dataset_stats,
            evaluate_test,
            model_create,
            model_delete,
            model_export,
            model_import,
            model_list,
            model_load,
            model_rename,
            train_bulk,
            train_one_epoch,
            training_history,
            training_reset,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
