def reward(obs, action, next_obs, terminated, truncated, info):
    x0 = float(obs[0]); y0 = float(obs[1]); vx0 = float(obs[2]); vy0 = float(obs[3])
    a0 = float(obs[4]); w0 = float(obs[5]); c10 = float(obs[6]); c20 = float(obs[7])
    x1 = float(next_obs[0]); y1 = float(next_obs[1]); vx1 = float(next_obs[2]); vy1 = float(next_obs[3])
    a1 = float(next_obs[4]); w1 = float(next_obs[5]); l11 = float(next_obs[6]); l21 = float(next_obs[7])

    # --- tuned PD landing controller, inline: which engine should fire now? ---
    angle_targ = max(-0.4, min(0.4, 0.5 * x0 + 1.0 * vx0))
    hover_targ = 0.7 * abs(x0)
    angle_todo = (angle_targ - a0) * 1.0 - w0 * 1.0
    hover_todo = (hover_targ - y0) * 0.5 - vy0 * 0.8
    if (c10 > 0.5) or (c20 > 0.5):
        angle_todo = 0.0
        hover_todo = -vy0 * 0.8
    want = 0
    if (hover_todo > abs(angle_todo)) and (hover_todo > 0.05):
        want = 2
    elif angle_todo < -0.05:
        want = 3
    elif angle_todo > 0.05:
        want = 1

    if int(action) == want:
        r = 1.5
    else:
        r = -0.5

    # --- light potential-based physical shaping ---
    d0 = math.sqrt(x0 * x0 + y0 * y0)
    d1 = math.sqrt(x1 * x1 + y1 * y1)
    s0 = math.sqrt(vx0 * vx0 + vy0 * vy0)
    s1 = math.sqrt(vx1 * vx1 + vy1 * vy1)
    pot0 = -100.0 * d0 - 100.0 * s0 - 100.0 * abs(a0) - 20.0 * abs(w0) + 10.0 * c10 + 10.0 * c20
    pot1 = -100.0 * d1 - 100.0 * s1 - 100.0 * abs(a1) - 20.0 * abs(w1) + 10.0 * l11 + 10.0 * l21
    r = r + 0.1 * (pot1 - pot0)
    r = r - 0.1

    if terminated:
        gentle = (s1 < 0.2) and (abs(a1) < 0.3) and ((l11 + l21) > 0.0)
        if gentle:
            r = r + 60.0 + 40.0 * max(0.0, 1.0 - abs(x1) / 0.2) + 20.0 * l11 * l21
        else:
            r = r - 60.0
    elif truncated:
        r = r - 40.0
    return float(r)
