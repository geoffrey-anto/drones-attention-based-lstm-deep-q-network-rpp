from src.Environment import Environment

if __name__ == "__main__":
    env = Environment(20, 20, 0.6)
    # print(f"Initial State: {env.agent.get_state()}")

    # # Target position
    # target_x, target_y, target_z = 5, 5, 0.6

    # # Move the drone in a curve
    # steps = 100  # Total steps to take
    # for i in range(steps):
    #     # Depending on the step, take left, right or straight actions
    #     if i % 10 < 5:  # Slight curve to the right
    #         env.agent.move_drone(2)  # Action 2 turns right
    #     else:  # Slight curve to the left
    #         env.agent.move_drone(0)  # Action 0 turns left

    #     # Print the drone's state every 10 steps to observe the curve
    #     if i % 10 == 0:
    #         print(f"Step {i}: {env.agent.get_state()}")
        
    #     # Check if we are close to the target position (simple stopping condition)
    #     x, y, z = env.agent.get_state()[:3]
    #     if abs(x - target_x) < 0.1 and abs(y - target_y) < 0.1:
    #         print(f"Drone reached close to the target at step {i}")
    #         break

    # # Final position
    # print(f"Final Position: {env.agent.get_state()}")
    
    for i in range(17):
        env.agent.move_drone(2) 
        
    for i in range(60):
        env.agent.move_drone(1)
        print(env.agent.get_state())
        
    # # for i in range(8):
    # #     env.agent.move_drone(0)
    
    # # for i in range(20):
    # #     env.agent.move_drone(1)
    
    env.show_environment()
