# Rewards being close to a point just above the pad.
def reward(obs, action, next_obs, terminated, truncated, info):
    x, y, vx, vy, angle, ang_vel, leg1, leg2 = next_obs
    dist = ((x - 0.0) ** 2 + (y - 1.0) ** 2) ** 0.5
    return 1.0 - min(1.0, dist)
