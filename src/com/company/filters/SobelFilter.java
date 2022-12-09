package com.company.filters;

import com.company.color_spaces.HSV;
import com.company.color_spaces.Lab;
import com.company.color_spaces.RGB;

public abstract class SobelFilter {
    protected static double kernel[][], kernelAdditional[][];
    protected static int N;
    public static double theta = 0., lambda = 2., psi = 0, gamma = 1.;

    public SobelFilter() {
    }

    public static void buildKernel() {
        N = 3;
        kernel = new double[3][3];
        kernel[0][0] = -1.;
        kernel[1][0] = -2.;
        kernel[2][0] = -1.;
        kernel[0][2] = 1.;
        kernel[1][2] = 2.;
        kernel[2][2] = 1.;

        kernelAdditional = new double[3][3];
        kernelAdditional[0][0] = -1.;
        kernelAdditional[0][1] = -2.;
        kernelAdditional[0][2] = -1.;
        kernelAdditional[2][0] = 1.;
        kernelAdditional[2][1] = 2.;
        kernelAdditional[2][2] = 1.;
    }

    public static double[][][] applyKernal(int width, int height, double[][][] matrix) {
        buildKernel();

        for (int X = 0; X < width; X++) {
            for (int Y = 0; Y < height; Y++) {
                RGB Gx=new RGB(0), Gy= new RGB(0);
                for (int i = X - N / 2, ii = 0; ii < N; i++, ii++) {
                    for (int j = Y - N / 2, jj = 0; jj < N; j++, jj++) {
                        int x = i;
                        int y = j;
                        if (!(i >= 0 && j >= 0 && i < width && j < height)) {
                            if (x < 0) x = 0;
                            if (x >= width) x = width - 1;

                            if (y < 0) y = 0;
                            if (y >= height) y = height - 1;
                        }
                        RGB rgb = new RGB(matrix[0][x][y], matrix[1][x][y], matrix[2][x][y]);

                        HSV hsv = new HSV(rgb);
                        hsv.changeS(-100);
                        rgb = hsv.toRGB();


                        Gx = Gx.addRGB(rgb.multiplyDouble(kernel[ii][jj]));
                        Gy = Gy.addRGB(rgb.multiplyDouble(kernelAdditional[ii][jj]));

                    }
                }
                RGB newRGB = (Gx.square().addRGB(Gy.square())).sqrt();
                matrix[0][X][Y] = newRGB.getR();
                matrix[1][X][Y] = newRGB.getG();
                matrix[2][X][Y] = newRGB.getB();
             //   matrix[0][X][Y] = Math.max(0., Math.min(255., matrix[0][X][Y]));
             //   matrix[1][X][Y] = Math.max(0., Math.min(255., matrix[1][X][Y]));
             //   matrix[2][X][Y] = Math.max(0., Math.min(255., matrix[2][X][Y]));
            }
        }

         double mn = 0, mx = 255;

       for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                mn = Math.min(Math.min(mn, matrix[2][x][y]), Math.min(matrix[0][x][y], matrix[1][x][y]));
                mx = Math.max(Math.max(mx, matrix[2][x][y]), Math.max(matrix[0][x][y], matrix[1][x][y]));
            }
        }

        //double bound = mx - (mx - mn) * 0.1;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                //if (matrix[0][x][y] < bound)
                //    matrix[0][x][y] = matrix[1][x][y] = matrix[2][x][y] = 0.;
                //else
                //    matrix[0][x][y] = matrix[1][x][y] = matrix[2][x][y] = 255.;
                matrix[0][x][y] = (matrix[0][x][y] - mn) * 255. / (mx - mn);
                matrix[1][x][y] = (matrix[1][x][y] - mn) * 255. / (mx - mn);
                matrix[2][x][y] = (matrix[2][x][y] - mn) * 255. / (mx - mn);
            }
        }
        return matrix;
    }

    public static void outKernel() {
        String text = "";
        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
                text += "          " + Double.toString(kernel[x][y]);
            }
            text += "\n";
        }
        System.out.print(text);
    }

}
