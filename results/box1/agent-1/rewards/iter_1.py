def reward(obs, action, next_obs, terminated, truncated, info):
    # unpack previous and current state
    px = float(obs[0]); py = float(obs[1])
    pvx = float(obs[2]); pvy = float(obs[3])
    pang = float(obs[4])
    pl1 = float(obs[6]); pl2 = float(obs[7])

    x = float(next_obs[0]); y = float(next_obs[1])
    vx = float(next_obs[2]); vy = float(next_obs[3])
    ang = float(next_obs[4]); angv = float(next_obs[5])
    l1 = float(next_obs[6]); l2 = float(next_obs[7])

    # potential-based shaping (classic LunarLander style)
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

    # fuel cost
    if action == 2:
        r = r - 0.30
    elif action == 1 or action == 3:
        r = r - 0.03

    # time penalty: discourage hovering / loitering
    r = r - 0.15

    # keep angular rate small
    r = r - 2.0 * abs(angv)

    if terminated:
        speed = math.sqrt(vx * vx + vy * vy)
        on_pad = abs(x) < 0.35
        gentle = speed < 0.5
        upright = abs(ang) < 0.35
        touched = (l1 + l2) >= 1.0
        if on_pad and gentle and upright and touched:
            r = r + 250.0
        else:
            r = r - 150.0
    return float(r)
