package com.company.filters;

import com.company.color_spaces.Lab;
import com.company.color_spaces.RGB;

public abstract class GaborFilter {
    protected static double kernel[][];
    protected static int N;
    public static double theta = getRadianFromDegree(45), lambda = 2., gamma = 1.;
    public static double psi = Math.PI / 3.8;

    public GaborFilter() {
    }

    private static double getRadianFromDegree(double angle) {
        return angle * Math.PI / 180.;
    }

    public static void buildKernel() {
        double p = 2. * Math.PI / lambda, p1 = -2. * Math.pow(0.56 * lambda, 2.);
        N = 5;
        kernel = new double[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double x = i - N / 2, y = j - N / 2;
                double x1 = x * Math.cos(theta) + y * Math.sin(theta), y1 = -x * Math.sin(theta) + y * Math.cos(theta);
                kernel[i][j] = Math.exp((x1 * x1 + Math.pow(gamma * y1, 2.)) / p1) * Math.cos(p * x1 + psi);
            }
        }
    }

    public static double[][][] applyKernal(int width, int height, double[][][] matrix) {
        buildKernel();
        //double answer[][][]=new double[1][width][height];
        for (int X = 0; X < width; X++) {
            for (int Y = 0; Y < height; Y++) {
                double value = 0;
                int x = 0, y = 0;
                for (int i = X - N / 2, ii = 0; ii < N; i++, ii++) {
                    for (int j = Y - N / 2, jj = 0; jj < N; j++, jj++) {
                        x = i;
                        y = j;
                        if (!(i >= 0 && j >= 0 && i < width && j < height)) {
                            if (x < 0) x = 0;
                            if (x >= width) x = width - 1;

                            if (y < 0) y = 0;
                            if (y >= height) y = height - 1;
                        }
                        RGB rgb = new RGB(matrix[0][x][y], matrix[1][x][y], matrix[2][x][y]);
                        value += new Lab(rgb).getL() * kernel[ii][jj];
                    }
                }
                //value = Math.max(0., Math.min(255., value));
                matrix[0][X][Y] = value;
                matrix[1][X][Y] = value;
                matrix[2][X][Y] = value;
                //answer[0][X][Y]=value;
            }
        }
        return matrix;
        //return answer;
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
