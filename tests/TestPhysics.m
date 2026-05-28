classdef TestPhysics < matlab.unittest.TestCase

    properties
        bt_params
    end

    methods (TestClassSetup)
        function setup(testCase)
            testCase.bt_params = { ...
                2.0, 1.0, 3.0, 1.0, 2.0, 0.5, ...
                1000.0, 9.81, 5.0, 2.0, ...
                0.8, 0.9, 0.5, 100.0, 1800.0, ...
                pi/6, 0.0, 0.0, 0.0};
        end
    end

    methods (Test)

        %% ---------------- blade interaction ----------------
        function testBladeZeroFill(testCase)
            [FT, Mb] = blade_terrain_interaction( ...
                1,1,1,1,0,1,0,1,1,0);
            testCase.verifyLessThanOrEqual(FT, 0);
        end

        function testBladeFinite(testCase)
            [FT, Mb] = blade_terrain_interaction( ...
                1,1,1,1,0,1,0,1,1,1);
            testCase.verifyTrue(isfinite(FT));
            testCase.verifyTrue(isfinite(Mb));
        end

        %% ---------------- get_x_icr ----------------
        function testICRZero(testCase)
            q = zeros(6,1);
            q_dot = zeros(6,1);
            val = get_x_icr(q, q_dot, 2);
            testCase.verifyEqual(val, 0);
        end

        function testICRClipping(testCase)
            q = zeros(6,1);
            q_dot = [0;10;0;0;0;1];
            val = get_x_icr(q, q_dot, 2);
            testCase.verifyLessThanOrEqual(abs(val),1);
        end

        %% ---------------- v_to_q_dot ----------------
        function testVtoQdotZero(testCase)
            q = zeros(6,1);
            v = zeros(2,1);
            out = v_to_q_dot(q,1,v);
            testCase.verifyEqual(out, zeros(6,1));
        end

        function testVtoQdotLinearity(testCase)
            q = zeros(6,1);
            v = [1;2];
            out1 = v_to_q_dot(q,1,v);
            out2 = v_to_q_dot(q,1,2*v);
            testCase.verifyEqual(out2, 2*out1, 'RelTol',1e-6);
        end

        %% ---------------- vel limiter ----------------
        function testVelLimiter(testCase)
            v = [2;-3];
            limit = [1;1];
            out = vel_limiter(v, limit);
            testCase.verifyEqual(out, [1;-1]);
        end

        function testVelLimiterNoReverse(testCase)
            v = [-2;1];
            limit = [3;3];
            out = vel_limiter(v, limit);
            testCase.verifyEqual(out(1), 0);
        end

        %% ---------------- hydraulics ----------------
        function testHydraulics(testCase)
            ang = [1;2;3];
            vel = [0.1;0.2;0.3];
            gain = 2;
        
            actual = hydraulics(ang,vel,gain);
            expected = ang + 2*vel;
        
            testCase.verifyEqual(actual(:), expected(:), 'RelTol',1e-6);
        end

        %% ---------------- controller ----------------
        function testControllerZero(testCase)
            ang = [0;0;0];
            des = [0;0;0];
        
            [err, ~] = controller_errors(ang,0,des,1);
        
            testCase.verifyEqual(err(:), zeros(3,1), 'AbsTol',1e-8);
        end

        function testControllerFinite(testCase)
            ang = [0;0.1;0];
            des = [0;1;0];
            [~, plot] = controller_errors(ang,0.5,des,1);
            testCase.verifyTrue(all(isfinite(plot)));
        end

    end
end