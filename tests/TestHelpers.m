classdef TestHelpers < matlab.unittest.TestCase

    methods (Test)
        
        %% ---------------- saturation ----------------
        function testSaturationBounds(testCase)
            testCase.verifyEqual(saturation(10,5), 5);
            testCase.verifyEqual(saturation(-10,5), -5);
            testCase.verifyEqual(saturation(3,5), 3);
        end

        function testSaturationSymmetry(testCase)
            x = 4.2; L = 3.0;
            testCase.verifyEqual(saturation(x,L), -saturation(-x,L));
        end

        %% ---------------- G ----------------
        function testGSliding(testCase)
            testCase.verifyEqual(G(10,5,1), -5);
            testCase.verifyEqual(G(10,5,-1), 5);
        end

        function testGStatic(testCase)
            testCase.verifyEqual(G(3,5,0), -3);
        end

        function testGBreakaway(testCase)
            testCase.verifyEqual(G(10,5,0), -5);
            testCase.verifyEqual(G(-10,5,0), 5);
        end

        %% ---------------- yc ----------------
        function testYcZero(testCase)
            testCase.verifyEqual(yc(0,0,10), 0.0);
        end

        function testYcSymmetry(testCase)
            testCase.verifyEqual(yc(5,5,10), 0.0, 'AbsTol',1e-8);
        end

        function testYcScaling(testCase)
            y1 = yc(2,4,10);
            y2 = yc(2,4,20);
            testCase.verifyEqual(y2, 2*y1, 'RelTol',1e-6);
        end

        %% ---------------- Hx ----------------
        function testHxMidpoint(testCase)
            testCase.verifyEqual(Hx(0,2,6,10), 4, 'AbsTol',1e-8);
        end

        function testHxEndpoints(testCase)
            B1 = 10;
            testCase.verifyEqual(Hx(-B1/2,2,6,B1), 2, 'AbsTol',1e-8);
            testCase.verifyEqual(Hx(B1/2,2,6,B1), 6, 'AbsTol',1e-8);
        end

        %% ---------------- Dx ----------------
        function testDxBasic(testCase)
            val = Dx(0,pi/4,2,6,10);
            testCase.verifyEqual(val, Hx(0,2,6,10), 'RelTol',1e-6);
        end

        function testDxMonotonic(testCase)
            yb = 1;
            val1 = Dx(yb,pi/6,2,6,10);
            val2 = Dx(yb,pi/3,2,6,10);
            testCase.verifyGreaterThan(val1, val2);
        end

    end
end