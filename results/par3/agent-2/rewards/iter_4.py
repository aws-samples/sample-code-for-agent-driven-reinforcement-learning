def reward(obs, action, next_obs, terminated, truncated, info):
    x0 = float(obs[0])
    y0 = float(obs[1])
    vx0 = float(obs[2])
    vy0 = float(obs[3])
    a0 = float(obs[4])
    w0 = float(obs[5])
    l10 = float(obs[6])
    l20 = float(obs[7])
    x1 = float(next_obs[0])
    y1 = float(next_obs[1])
    vx1 = float(next_obs[2])
    vy1 = float(next_obs[3])
    a1 = float(next_obs[4])
    w1 = float(next_obs[5])
    l11 = float(next_obs[6])
    l21 = float(next_obs[7])

    speed0 = math.sqrt(vx0 * vx0 + vy0 * vy0)
    speed1 = math.sqrt(vx1 * vx1 + vy1 * vy1)

    # Potential-based shaping: over the pad, upright, inside a descent corridor.
    pot0 = (
        -100.0 * math.sqrt(x0 * x0 + y0 * y0)
        - 100.0 * speed0
        - 40.0 * abs(x0)
        - 100.0 * abs(a0)
        - 20.0 * abs(w0)
        - 60.0 * max(0.0, abs(vy0) - 0.12 - 0.40 * abs(y0))
        + 10.0 * (l10 + l20)
    )
    pot1 = (
        -100.0 * math.sqrt(x1 * x1 + y1 * y1)
        - 100.0 * speed1
        - 40.0 * abs(x1)
        - 100.0 * abs(a1)
        - 20.0 * abs(w1)
        - 60.0 * max(0.0, abs(vy1) - 0.12 - 0.40 * abs(y1))
        + 10.0 * (l11 + l21)
    )
    r = pot1 - pot0

    # Fuel cost.
    if action == 2:
        r = r - 0.20
    elif action == 1 or action == 3:
        r = r - 0.03

    # Mild anti-hover.
    if (l11 + l21) == 0.0 and y1 < 0.7 and abs(vy1) < 0.06:
        r = r - 0.25
    r = r - 0.03

    if terminated:
        # Judge the touchdown on the velocity BEFORE the contact impulse,
        # because a collision zeroes the post-step velocity.
        impact = speed0
        r = r + 180.0 - 320.0 * impact
        r = r - 120.0 * abs(a1) - 40.0 * abs(w0)
        if impact < 0.4:
            r = r + 70.0 * max(0.0, 1.0 - abs(x1) / 0.30)
    elif truncated:
        r = r - 60.0

    return r
