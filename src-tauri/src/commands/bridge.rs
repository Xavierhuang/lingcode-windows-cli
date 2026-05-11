use tauri::{AppHandle, State};
use std::collections::HashMap;
use crate::services::{BridgeService, ProviderService};
use crate::app_config::AppType;
use std::path::PathBuf;

#[tauri::command]
pub async fn run_agent_bridge(
    app: AppHandle,
    state: State<'_, crate::store::AppState>,
    prompt: String,
    provider_id: Option<String>,
    cwd: Option<String>
) -> Result<String, String> {
    // 1. Resolve provider
    let provider_service = ProviderService;
    let app_type = AppType::Claude; // Default to Claude for bridge
    
    let pid = match provider_id {
        Some(id) => id,
        None => {
            // Get active provider for Claude
            let global_config = state.get_global_config().await
                .map_err(|e| format!("Failed to get global config: {e}"))?;
            global_config.active_providers.get(&app_type.to_string())
                .cloned()
                .ok_or_else(|| "No active provider for Claude".to_string())?
        }
    };

    let providers = ProviderService::list(state.inner(), app_type.clone())
        .map_err(|e| format!("Failed to list providers: {e}"))?;
    
    let provider = providers.get(&pid)
        .ok_or_else(|| format!("Provider {} not found", pid))?;

    // 2. Extract environment variables from provider config
    // (This logic should ideally be shared with open_provider_terminal)
    let mut env_vars = HashMap::new();
    if let Some(obj) = provider.settings_config.as_object() {
        if let Some(env) = obj.get("env").and_then(|v| v.as_object()) {
            for (key, value) in env {
                if let Some(str_val) = value.as_str() {
                    env_vars.insert(key.clone(), str_val.to_string());
                }
            }
        }
    }

    // 3. Resolve CWD
    let launch_cwd = cwd.map(PathBuf::from);

    // 4. Spawn bridge
    BridgeService::spawn_bridge(
        &app,
        &prompt,
        &provider.name,
        launch_cwd,
        env_vars
    ).map_err(|e| format!("Failed to spawn agent bridge: {e}"))?;

    Ok(format!("Agent bridge spawned for provider {}", provider.name))
}
