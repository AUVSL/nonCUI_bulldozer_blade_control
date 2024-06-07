# nonCUI_bulldozer_blade_control
Code associated with a 3 degree of freedom (dof) bulldozer blade controller.

In order to run the code:
1) ```git clone https://github.com/AUVSL/nonCUI_bulldozer_blade_control ```
2) run main.m

Tool Boxes:
1) Simulink R2023a
2) Fuzzy (you can replace the fuzzy controllers and add PIDs or something else if you don't have this toolbox.)

Primary Files:
1) Parameters.m sets the soil and dozer parameters which are loaded in main.
2) Main.m sends the control commands and formatted parameters to the simulation files. After the simulation stops plot are generated.
3) Simulation_3d.slx is the simulation file where the varaibles are updated based on the forwards dynamics vomputed in the 'vehicle_model' block.
4) Errors_and_plots.m provides postion, orientation, and error plots that the author found helpful when debugging the dynamics.

Design Considerations:
1) The varaible names are less desciptive than what is best best practise for most code. This choice was made to better reflect the equation in the paper assiated with this code.
2) The PID in simulink resents the hyrolics and how they take time to hit a certain position. The was chosen arbitarly so the control plots in my paper looked nice for my stopping distance of 0.1 meters.
3) The x coordinate of the Instantaneous Center of Rotation was satured in line with "Path Tracking Control of Tracked Vehicles ~M. Ahmadi, V. Polotski, and R. Hurteau, 2000."
4) The code is written in accordance with the vehicle moving forward. If you make the vehicle backing up please make the blade force zero.
5) The driving force is really low to the point the vehicle cannot turn. This was done to make the forces in line with the sandy loam soil parameters the author, Sam Dekhterman, found in the literature. You will need need to increase the torque commands by at least x10 the base values to effectly turn.
6) The block strucure is a little odd. Partically how the track control and initalizaion are set at once in the 'track_control_and_stopping' block. Feel free to alter this structure when modifying/ porting the simulation code.
7) Lastly, please be carefull with the time step in simulink. If it is too large you will see oscillation in the integrators and thus postion. For that reason the ode4 Runge-Kutta solver with a step size of 0.0001 was used. Using a smaller step size dramatlly increased the run time. 
