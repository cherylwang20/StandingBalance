import opensim as osim

# Charger le modèle
model = osim.Model("Lumbar_C_210.osim")
state = model.initSystem()

# Charger les activations musculaires depuis le fichier .sto
muscle_activations = osim.Storage("InputFiles/30deg_Flex_motion.sto")

# Configurer l'outil ForwardTool
tool = osim.ForwardTool()
tool.setModel(model)
tool.setInitialTime(muscle_activations.getFirstTime())
tool.setFinalTime(muscle_activations.getLastTime())

# Sauvegarder le ControlSet et exécuter la simulation
control_set_file = "Outputs/ControlSet.xml"
control_set = osim.ControlSet(muscle_activations, 0, 0)
control_set.printToXML(control_set_file)
tool.setControlsFileName(control_set_file)

# Exécuter la simulation
tool.run()
