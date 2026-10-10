obs = obslua

local base_dir = script_path()
local overlay_launcher = base_dir .. "launch_overlay.sh"
local overlay_pid_file = base_dir .. "overlay.pid"
local overlay_log_file = base_dir .. "overlay.log"
local mask_file = base_dir .. "detection_mask.png"
local mask_state_file = base_dir .. "mask_state.json"
local filter_name = "旧UIの検出マスク（自動同期）"
local camera_source = "メディア"
local last_mask_revision = nil
local exiting = false

local function shell_quote(value)
    return "'" .. string.gsub(value, "'", "'\\''") .. "'"
end

local function overlay_process_alive()
    local handle = io.open(overlay_pid_file, "r")
    if handle == nil then return false end
    local pid = handle:read("*l")
    handle:close()
    if pid == nil or string.match(pid, "^%d+$") == nil then return false end
    local result = os.execute("/bin/ps -p " .. pid .. " -o command= | /usr/bin/grep -Fq "
        .. shell_quote(base_dir .. "latest_meteor_gif.py"))
    return result == true or result == 0
end

local function stop_overlay()
    local command = "pid=$(cat " .. shell_quote(overlay_pid_file) .. " 2>/dev/null || true); "
        .. "if [ -n \"$pid\" ] && /bin/ps -p \"$pid\" -o command= | /usr/bin/grep -Fq "
        .. shell_quote(base_dir .. "latest_meteor_gif.py") .. "; then "
        .. "/bin/kill \"$pid\" 2>/dev/null || true; fi; /bin/rm -f " .. shell_quote(overlay_pid_file)
    os.execute("/bin/sh -c " .. shell_quote(command))
end

local function start_overlay()
    if overlay_process_alive() then return end
    os.execute("/bin/bash " .. shell_quote(overlay_launcher))
end

local function sync_camera_mask()
    local handle = io.open(mask_state_file, "r")
    if handle == nil then return end
    local state = obs.obs_data_create_from_json(handle:read("*a"))
    handle:close()
    if state == nil then return end
    local revision = obs.obs_data_get_string(state, "revision")
    local enabled = obs.obs_data_get_bool(state, "enabled")
    local source = obs.obs_get_source_by_name(camera_source)
    if source ~= nil then
        local filter = obs.obs_source_get_filter_by_name(source, filter_name)
        if filter == nil or revision ~= last_mask_revision then
            local settings = obs.obs_data_create()
            obs.obs_data_set_string(settings, "image_path", mask_file)
            -- Multiply by a binary RGB image: suppressed pixels become black,
            -- exactly as in the legacy preview. This precedes scene cropping.
            obs.obs_data_set_string(settings, "type", "blend_mul_filter.effect")
            obs.obs_data_set_int(settings, "color", 0xFFFFFF)
            obs.obs_data_set_double(settings, "opacity", 1.0)
            obs.obs_data_set_bool(settings, "stretch", true)
            if filter == nil then
                filter = obs.obs_source_create_private("mask_filter_v2", filter_name, settings)
                if filter ~= nil then obs.obs_source_filter_add(source, filter) end
            else
                obs.obs_source_update(filter, settings)
            end
            obs.obs_data_release(settings)
            if filter ~= nil then
                last_mask_revision = revision
                obs.obs_frontend_save()
            end
        end
        if filter ~= nil then
            obs.obs_source_set_enabled(filter, enabled)
            obs.obs_source_release(filter)
        end
        obs.obs_source_release(source)
    end
    obs.obs_data_release(state)
end

local function check_overlay()
    if exiting then return end
    start_overlay()
    sync_camera_mask()
end

function on_event(event)
    if event == obs.OBS_FRONTEND_EVENT_EXIT then
        exiting = true
        stop_overlay()
    end
end

function script_description()
    return "旧UIが保存した検出マスクをカメラ映像と流星GIFに自動同期します。OBSを開いている間はプレビューでも更新し、OBS終了時に停止します。"
end

function script_defaults(settings)
    obs.obs_data_set_default_string(settings, "camera_source", "メディア")
end

function script_properties()
    local properties = obs.obs_properties_create()
    obs.obs_properties_add_text(properties, "camera_source", "カメラ映像のソース名", obs.OBS_TEXT_DEFAULT)
    return properties
end

function script_update(settings)
    local selected = obs.obs_data_get_string(settings, "camera_source")
    if selected ~= "" and selected ~= camera_source then
        local previous = obs.obs_get_source_by_name(camera_source)
        if previous ~= nil then
            local filter = obs.obs_source_get_filter_by_name(previous, filter_name)
            if filter ~= nil then
                obs.obs_source_set_enabled(filter, false)
                obs.obs_source_release(filter)
            end
            obs.obs_source_release(previous)
        end
        camera_source = selected
        last_mask_revision = nil
    end
end

function script_load(settings)
    script_update(settings)
    obs.obs_frontend_add_event_callback(on_event)
    obs.timer_add(check_overlay, 2000)
    check_overlay()
    obs.obs_frontend_save()
end

function script_unload()
    exiting = true
    obs.timer_remove(check_overlay)
    obs.obs_frontend_remove_event_callback(on_event)
    stop_overlay()
end
