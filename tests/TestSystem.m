classdef TestSystem < matlab.unittest.TestCase

    methods (Test)

        %% ---------------- pipeline consistency ----------------
        function testPipelineFinite(testCase)
            q = zeros(6,1);
            q_dot = zeros(6,1);

            x_icr = get_x_icr(q, q_dot, 2);
            v = [1; 0.1];

            q_dot_out = v_to_q_dot(q, x_icr, v);

            testCase.verifyTrue(all(isfinite(q_dot_out)));
        end

        %% ---------------- symmetry ----------------
        function testSymmetry(testCase)
            q = zeros(6,1);
            v = [1;0];

            out = v_to_q_dot(q, 1e6, v);

            testCase.verifyEqual(out(2), 0, 'AbsTol',1e-8);
        end

    end
end