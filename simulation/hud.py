import pygame
from settings import *

def draw_hud(surface, signal_controller, initial_counts, gst_values, active_vehicles, cycle_count=0):
    """
    Renders a stunning glassmorphic HUD panel in the top-left corner.
    Displays signal status, computed GST values, vehicle load profiles, and live simulation stats.
    Uses real-world road names from Prithivi Chowk, Pokhara.
    """
    # 1. Create a translucent glassmorphic surface
    hud_width = 480
    hud_height = 340
    hud_surf = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
    
    # Fill with semi-transparent dark slate blue
    hud_surf.fill((15, 23, 42, 220))
    # Draw slate border
    pygame.draw.rect(hud_surf, (71, 85, 105, 255), (0, 0, hud_width, hud_height), 2)
    
    # Subtle glowing title bar
    pygame.draw.rect(hud_surf, (30, 41, 59, 255), (0, 0, hud_width, 40))
    pygame.draw.line(hud_surf, (71, 85, 105, 255), (0, 40), (hud_width, 40), 1)

    # 2. Set up modern system fonts
    try:
        font_title = pygame.font.SysFont('Segoe UI', 16, bold=True)
        font_body = pygame.font.SysFont('Segoe UI', 13, bold=True)
        font_data = pygame.font.SysFont('Consolas', 13)
        font_glowing = pygame.font.SysFont('Segoe UI', 15, bold=True)
    except:
        font_title = pygame.font.SysFont('Arial', 16, bold=True)
        font_body = pygame.font.SysFont('Arial', 13, bold=True)
        font_data = pygame.font.SysFont('Arial', 13)
        font_glowing = pygame.font.SysFont('Arial', 15, bold=True)

    # Title text
    title_text = font_title.render("AI TRAFFIC SIGNAL CONTROLLER", True, (241, 245, 249))
    hud_surf.blit(title_text, (20, 10))

    y_offset = 55

    # 3. Active Phase Timing
    curr_idx = signal_controller.get_green_direction()
    curr_name = signal_controller.direction_names[curr_idx]
    remaining = signal_controller.timer

    phase_label = font_body.render("CURRENT PHASE:", True, (148, 163, 184))
    hud_surf.blit(phase_label, (20, y_offset))
    
    rn_full = ROAD_NAMES[curr_name]
    phase_str = f"{rn_full} [GREEN] | {remaining:.1f}s left"
    phase_val = font_glowing.render(phase_str, True, COLOR_GREEN)
    hud_surf.blit(phase_val, (135, y_offset - 2))
    
    y_offset += 25

    # Show orange preview info when <= 5s remaining
    next_idx = (curr_idx + 1) % 4
    next_name = signal_controller.direction_names[next_idx]
    next_state = signal_controller.get_signal_state(next_idx)
    if next_state == 'ORANGE':
        rn_next = ROAD_NAMES[next_name]
        orange_str = f"{rn_next} [ORANGE] - Get Ready!"
        orange_val = font_data.render(orange_str, True, COLOR_YELLOW)
        hud_surf.blit(orange_val, (135, y_offset))
    y_offset += 22

    # 4. Live / Pre-computed GST values
    gst_label = font_body.render("GST SYSTEM VALUES:", True, (148, 163, 184))
    hud_surf.blit(gst_label, (20, y_offset))
    y_offset += 20

    # Draw a table-like view for GSTs
    for i in range(4):
        rn = ROAD_NAMES_SHORT[DIRECTIONS[i]]
        gst_val = gst_values[i]
        
        sig_state = signal_controller.get_signal_state(i)
        if sig_state == 'GREEN':
            text_color = COLOR_GREEN
        elif sig_state == 'ORANGE':
            text_color = COLOR_YELLOW
        else:
            text_color = (241, 245, 249)
        
        gst_text = font_data.render(f"  {rn}: {gst_val:.2f} seconds", True, text_color)
        hud_surf.blit(gst_text, (20, y_offset))
        y_offset += 16

    y_offset += 10

    # 5. Vehicle Load Profiling (Inputs)
    load_label = font_body.render("VEHICLE DETECTIONS PER APPROACH:", True, (148, 163, 184))
    hud_surf.blit(load_label, (20, y_offset))
    y_offset += 20

    for i in range(4):
        sd = DIRECTIONS[i]
        counts = initial_counts.get(sd, {})
        count_strs = []
        for vt in ['car', 'motorcycle', 'truck', 'bus']:
            cnt = counts.get(vt, 0)
            if cnt > 0:
                short_vt = vt[0].upper() + vt[1:3]
                count_strs.append(f"{short_vt}:{cnt}")
        
        rn_short = ROAD_NAMES_SHORT[sd]
        load_profile = f"  {rn_short}: " + (", ".join(count_strs) if count_strs else "No traffic")

        sig_state = signal_controller.get_signal_state(i)
        if sig_state == 'GREEN':
            load_color = COLOR_GREEN
        elif sig_state == 'ORANGE':
            load_color = COLOR_YELLOW
        else:
            load_color = (226, 232, 240)
        
        load_text = font_data.render(load_profile, True, load_color)
        hud_surf.blit(load_text, (20, y_offset))
        y_offset += 16

    # 6. Blit the HUD onto main surface
    surface.blit(hud_surf, (15, 15))

    # Add a separate simulation stats indicator in the top-right corner
    stats_width = 320
    stats_height = 60
    stats_surf = pygame.Surface((stats_width, stats_height), pygame.SRCALPHA)
    stats_surf.fill((15, 23, 42, 220))
    pygame.draw.rect(stats_surf, (71, 85, 105, 255), (0, 0, stats_width, stats_height), 2)
    
    active_cnt = len(active_vehicles)
    stats_text = font_body.render(f"ACTIVE VEHICLES IN JUNCTION: {active_cnt}", True, (241, 245, 249))
    stats_surf.blit(stats_text, (12, 8))

    cycle_text = font_body.render(f"CYCLE COMPLETED: {cycle_count}", True, (16, 185, 129))
    stats_surf.blit(cycle_text, (12, 32))
    surface.blit(stats_surf, (SCREEN_WIDTH - stats_width - 15, 15))