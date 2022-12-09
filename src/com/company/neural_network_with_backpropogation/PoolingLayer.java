package com.company.neural_network_with_backpropogation;

import com.company.neural_network.Size;

public final class PoolingLayer implements BackpropogationLayer {
    private Size inputSize, outputSize;
    //private final int WEIGHT = 0, DELTA = 1;
    private int kernelSize;
    private int xOfLastMax[][][], yOfLastMax[][][];

    public PoolingLayer(Size inputSize, int kernelSize) {
        if (!inputSize.isPositive() || kernelSize < 3 || kernelSize % 2 == 0)
            throw new IllegalArgumentException();
        this.inputSize = inputSize;
        this.kernelSize = kernelSize;
        this.outputSize = new Size(inputSize.depth, inputSize.width - kernelSize + 1, inputSize.height - kernelSize + 1);
        xOfLastMax = new int[outputSize.depth][outputSize.width][outputSize.height];
        yOfLastMax = new int[outputSize.depth][outputSize.width][outputSize.height];
    }

    @Override
    public Size getInputSize() {
        return inputSize;
    }

    @Override
    public Size getSize() {
        return outputSize;
    }

    @Override
    public double[][][] computeOutput(double[][][] input) {
        if (input == null || !inputSize.compare(new Size(input)))
            throw new IllegalArgumentException();

        double[][][] output = new double[outputSize.depth][outputSize.width][outputSize.height];
        for (int d = 0; d < outputSize.depth; d++) {
            for (int x2 = 0; x2 < outputSize.width; x2++) {
                for (int y2 = 0; y2 < outputSize.height; y2++) {
                    output[d][x2][y2] = Double.MIN_VALUE;
                    for (int i = 0, x1 = x2; i < Math.min(inputSize.width - x2, kernelSize); i++, x1++) {
                        for (int j = 0, y1 = y2; j < Math.min(inputSize.width - y2, kernelSize); j++, y1++) {
                            if (output[d][x2][y2] < input[d][x1][y1]) {
                                output[d][x2][y2] = input[d][x1][y1];
                                xOfLastMax[d][x2][y2] = x1;
                                yOfLastMax[d][x2][y2] = y1;
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    @Override
    public void randomize(double min, double max) {
    }

    @Override
    public double[][][] computeBackwardError(double[][][] input, double[][][] error) {
        if (input == null || !getInputSize().compare(new Size(input)) || error == null || error.length != outputSize.depth)
            throw new IllegalArgumentException();

        //double[][][] output = computeOutput(input);
        double[][][] backwardError = new double[inputSize.depth][inputSize.width][inputSize.height];
        for (int d = 0; d < inputSize.depth; d++) {
            for (int x1 = 0; x1 < inputSize.width; x1++) {
                for (int y1 = 0; y1 < inputSize.height; y1++) {
                    backwardError[d][x1][y1] = 0;
                    {
                        for (int x2 = Math.max(0, x1 - kernelSize + 1), i = kernelSize - 1 - x1 + x2; x2 < Math.min(x1, outputSize.width); x2++, i++) {
                            for (int y2 = Math.max(0, y1 - kernelSize + 1), j = kernelSize - 1 - y1 + y2; y2 < Math.min(y1, outputSize.height); y2++, j++) {
                                if (xOfLastMax[d][x2][y2] == x1 && yOfLastMax[d][x2][y2] == y1)
                                    backwardError[d][x1][y1] += error[d][x2][y2];
                            }
                        }
                    }
                }
            }
        }
        return backwardError;
    }

    @Override
    public void adjust(double[][][] input, double[][][] error, double rate, double momentum) {
        if (input == null || !getInputSize().compare(new Size(input)) || error == null || error.length != outputSize.depth)
            throw new IllegalArgumentException();

        /*double[][][] output = computeOutput(input);

        for (int d = 0; d < outputSize.depth; d++) {
            for (int x2 = 0; x2 < outputSize.width; x2++) {
                for (int y2 = 0; y2 < outputSize.height; y2++) {
                    final double gradient = error[d][x2][y2];
                    for (int i = 0, x1 = x2; i < Math.min(inputSize.width - x2, kernelSize); i++, x1++) {
                        for (int j = 0, y1 = y2; j < Math.min(inputSize.height - y2, kernelSize); j++, y1++) {
                            weights[k][i][j][DELTA] = (1. - momentum) * rate * input[d1][x1][y1] * gradient + momentum * weights[k][i][j][DELTA];
                            weights[k][i][j][WEIGHT] += weights[k][i][j][DELTA];
                        }
                    }
                }
            }
        }*/
    }

}
