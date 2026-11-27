def reward(obs, action, next_obs, terminated, truncated, info):
    px = float(obs[0]); py = float(obs[1])
    pvx = float(obs[2]); pvy = float(obs[3])
    pang = float(obs[4])
    pl1 = float(obs[6]); pl2 = float(obs[7])

    x = float(next_obs[0]); y = float(next_obs[1])
    vx = float(next_obs[2]); vy = float(next_obs[3])
    ang = float(next_obs[4]); angv = float(next_obs[5])
    l1 = float(next_obs[6]); l2 = float(next_obs[7])

    h = max(y, 0.0)
    contact = (l1 + l2) >= 1.0
    speed = math.sqrt(vx * vx + vy * vy)

    prev_pot = (
        -100.0 * math.sqrt(px * px + py * py)
        - 100.0 * math.sqrt(pvx * pvx + pvy * pvy)
        - 100.0 * abs(pang)
        + 10.0 * pl1
        + 10.0 * pl2
    )
    pot = (
        -100.0 * math.sqrt(x * x + y * y)
        - 100.0 * math.sqrt(vx * vx + vy * vy)
        - 100.0 * abs(ang)
        + 10.0 * l1
        + 10.0 * l2
    )
    r = pot - prev_pot

    if contact:
        if action != 0:
            r = r - 3.0
        r = r - 8.0 * speed - 2.0 * abs(angv)
        if (l1 + l2) < 2.0:
            r = r - 1.5
        r = r - 2.0 * abs(ang)
    else:
        r = r + 1.0 * min(1.0, h / 0.5)
        sink_t = 0.15 + 0.45 * h
        r = r - 4.0 * abs(-vy - sink_t)
        # lateral capture: tilting to close the offset is cheap, drifting is not
        vx_t = max(-0.5, min(0.5, -0.8 * x))
        r = r - 4.0 * abs(vx - vx_t)
        r = r - 1.2 * abs(x)
        if h < 0.30:
            r = r - 1.5 * abs(ang) - 0.5 * abs(angv)
            r = r - 5.0 * max(0.0, speed - 0.30) - 4.0 * abs(x)
        else:
            r = r - 0.5 * abs(ang) - 0.3 * abs(angv)
        if action == 2:
            r = r - 0.10
        elif action == 1 or action == 3:
            r = r - 0.02

    if terminated:
        q_speed = max(0.0, 1.0 - speed / 0.8)
        q_pos = max(0.0, 1.0 - abs(x) / 0.6)
        q_ang = max(0.0, 1.0 - abs(ang) / 0.6)
        r = r - 30.0 + 320.0 * q_speed * q_pos * q_ang
        q_pad = max(0.0, 1.0 - abs(x) / 0.22)
        r = r + 250.0 * q_pad * q_speed * q_ang
    elif truncated:
        r = r - 80.0
    return float(r)
