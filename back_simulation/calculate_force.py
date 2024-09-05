import mujoco
import numpy as np

def calculate_force(moment, P1, P2, P3):
    r = P2 - P1
    d = P3 - P2
    d_normalized = d / np.linalg.norm(d)
    magnitude_F = np.dot(moment, np.cross(r, d_normalized)) / np.linalg.norm(np.cross(r, d_normalized))
    force = magnitude_F * d_normalized
    return force

def get_attachment_points(model, data, tendon_id):
    """
    Retrieve the coordinates of the attachment points for a given tendon.
    """
    # Get the site IDs connected by the tendon
    site1_id = model.tendon_site[tendon_id, 0]
    site2_id = model.tendon_site[tendon_id, 1]

    # Get the world positions of these sites
    P1 = data.site_xpos[site1_id]
    P2 = data.site_xpos[site2_id]

    return P1, P2

def compute_muscle_force(model, data, actuator_id, tendon_id):
    """
    Compute the force exerted by a muscle given its actuator moment and attachment points.
    """
    # Retrieve the moment applied by the actuator
    moment = data.actuator_moment[:, actuator_id]

    # Retrieve the attachment points of the muscle (tendon)
    P1, P2 = get_attachment_points(model, data, tendon_id)

    # Define a third point P3 to calculate the direction of the force
    # P3 can be chosen as a point along the tendon or a reference point in space
    P3 = P2 + np.array([0, 0, 1])  # Example: a point directly above P2

    # Calculate the force
    force = calculate_force(moment, P1, P2, P3)

    return force



# Example usage with MuJoCo
model = mujoco.MjModel.from_xml_path('C:/Users/morga/MyoBack/myosuite/myosuite/simhive/myo_sim/back/myobacklegs-Exoskeleton.xml')
data = mujoco.MjData(model)

# Assuming a simulation step has been run
mujoco.mj_step(model, data)

# Example IDs for the actuator and tendon
actuator_id = 0  # Replace with your actuator ID
tendon_id = 0  # Replace with your tendon ID

# Compute the force
force = compute_muscle_force(model, data, actuator_id, tendon_id)

print("The force exerted by the muscle is:", force)

