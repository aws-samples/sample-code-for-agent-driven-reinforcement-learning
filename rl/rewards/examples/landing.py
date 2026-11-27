# Potential-based shaping (closer, slower, upright, legs down), fuel costs, and a terminal
# landing or crash payoff. Potentials are inlined because the guard allows no helper functions.
def reward(obs, action, next_obs, terminated, truncated, info):
    x0, y0, vx0, vy0, a0, av0, l10, l20 = obs
    x1, y1, vx1, vy1, a1, av1, l11, l21 = next_obs
    s0 = (-100.0 * ((x0 * x0 + y0 * y0) ** 0.5) - 100.0 * ((vx0 * vx0 + vy0 * vy0) ** 0.5)
          - 100.0 * abs(a0) + 10.0 * l10 + 10.0 * l20)
    s1 = (-100.0 * ((x1 * x1 + y1 * y1) ** 0.5) - 100.0 * ((vx1 * vx1 + vy1 * vy1) ** 0.5)
          - 100.0 * abs(a1) + 10.0 * l11 + 10.0 * l21)
    r = s1 - s0
    if action == 2:
        r -= 0.30
    elif action == 1 or action == 3:
        r -= 0.03
    if terminated:
        landed = l11 == 1.0 and l21 == 1.0 and abs(vx1) < 0.3 and abs(vy1) < 0.3
        r += 100.0 if landed else -100.0
    return r
