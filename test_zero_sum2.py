gamma = 0.95
for realism in [1.0, 0.0]:
    for accuracy in [1.0, 0.0]:
        desired_inj_return = realism - accuracy
        r_ass = accuracy
        
        # We subtract gamma * r_ass to exactly cancel the future reward from GAE!
        r_inj = desired_inj_return - gamma * r_ass
        
        actual_inj_return = r_inj + gamma * r_ass
        
        print(f"R:{realism} A:{accuracy} -> r_inj:{r_inj:.2f}, G_inj:{actual_inj_return:.2f} (Desired:{desired_inj_return})")
