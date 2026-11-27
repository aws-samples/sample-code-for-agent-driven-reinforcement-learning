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

    # Purely potential-based shaping (telescoping => cannot be farmed by loitering).
    prev_pot = (
        -100.0 * math.sqrt(x0 * x0 + y0 * y0)
        - 100.0 * math.sqrt(vx0 * vx0 + vy0 * vy0)
        - 100.0 * abs(a0)
        - 20.0 * abs(w0)
        + 15.0 * l10
        + 15.0 * l20
        - 60.0 * max(0.0, -vy0) * max(0.0, 1.2 - y0)
    )
    pot = (
        -100.0 * math.sqrt(x1 * x1 + y1 * y1)
        - 100.0 * math.sqrt(vx1 * vx1 + vy1 * vy1)
        - 100.0 * abs(a1)
        - 20.0 * abs(w1)
        + 15.0 * l11
        + 15.0 * l21
        - 60.0 * max(0.0, -vy1) * max(0.0, 1.2 - y1)
    )
    r = pot - prev_pot

    # fuel cost: also pushes the agent to cut engines once it is down
    if action == 2:
        r -= 0.30
    elif action == 1 or action == 3:
        r -= 0.03

    # time cost: settle, do not hover
    r -= 0.10

    if terminated:
        speed = math.sqrt(vx1 * vx1 + vy1 * vy1)
        # The env only ends non-destructively when the lander has come to rest:
        # both legs down, essentially zero velocity, upright, over the pad.
        at_rest = (
            l11 > 0.0
            and l21 > 0.0
            and speed < 0.12
            and abs(w1) < 0.2
            and abs(a1) < 0.2
            and abs(x1) < 0.5
        )
        if at_rest:
            r += 300.0
        else:
            r -= 250.0
    elif truncated:
        r -= 150.0

    return r
