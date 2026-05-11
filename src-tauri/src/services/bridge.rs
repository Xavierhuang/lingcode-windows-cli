use std::path::PathBuf;
use std::process::{Command, Stdio};
use tauri::{AppHandle, Manager};
use crate::error::AppError;

pub struct BridgeService;

impl BridgeService {
    pub fn get_bridge_path(app: &AppHandle) -> Result<PathBuf, AppError> {
        let resource_dir = app.path().resource_dir()
            .map_err(|e| AppError::Config(format!("Failed to get resource dir: {e}")))?;
        
        let bridge_path = resource_dir.join("resources/agent-bridge/bridge.mjs");
        if !bridge_path.exists() {
            return Err(AppError::Config(format!("Bridge not found at {}", bridge_path.display())));
        }
        Ok(bridge_path)
    }

    pub fn spawn_bridge(
        app: &AppHandle,
        prompt: &str,
        provider: &str,
        cwd: Option<PathBuf>,
        env_vars: std::collections::HashMap<String, String>
    ) -> Result<(), AppError> {
        let bridge_path = Self::get_bridge_path(app)?;
        let node_path = "node"; // Assume node is on PATH for now

        let mut child = Command::new(node_path);
        child.arg(bridge_path);
        
        if let Some(c) = cwd {
            child.current_dir(c);
        }

        // Set up environment
        child.envs(env_vars);
        child.env("LINGCODE_PROMPT", prompt);
        child.env("LINGCODE_PROVIDER", provider);

        // In a real implementation, we would handle stdin/stdout/stderr
        // and bridge them to the Tauri frontend or terminal.
        child.stdout(Stdio::inherit())
             .stderr(Stdio::inherit());

        let mut _handle = child.spawn()
            .map_err(|e| AppError::IoContext {
                context: "Failed to spawn node bridge".to_string(),
                source: e
            })?;

        Ok(())
    }
}
