package com.company.neural_network_with_backpropogation;

import com.company.neural_network.Size;

public final class ConvolutionalLayer implements BackpropogationLayer {
    private Size inputSize, outputSize;
    private double[][][][] weights;
    private final int WEIGHT = 0, DELTA = 1;
    private boolean bipolar;
    private int kernelSize, increase;

    public ConvolutionalLayer(Size inputSize, int kernelSize, int increase, boolean bipolar) {
        if (!inputSize.isPositive() || kernelSize < 3 || kernelSize % 2 == 0)
            throw new IllegalArgumentException();
        this.inputSize = inputSize;
        this.kernelSize = kernelSize;
        this.increase = increase;
        this.outputSize = new Size(inputSize.depth * increase, inputSize.width - kernelSize + 1, inputSize.height - kernelSize + 1);
        weights = new double[increase][kernelSize][kernelSize][2];
        this.bipolar = bipolar;
    }

    public ConvolutionalLayer(Size inputSize, int kernelSize, int increase) {
        this(inputSize, kernelSize, increase, false);
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
        for (int d2 = 0; d2 < outputSize.depth; d2++) {
            int d1 = d2 / increase, k = d2 % increase;
            for (int x2 = 0; x2 < outputSize.width; x2++) {
                for (int y2 = 0; y2 < outputSize.height; y2++) {
                    for (int i = 0, x1 = x2; i < Math.min(inputSize.width - x2, kernelSize); i++, x1++) {
                        for (int j = 0, y1 = y2; j < Math.min(inputSize.width - y2, kernelSize); j++, y1++) {
                            output[d2][x2][y2] += input[d1][x1][y1] * weights[k][i][j][WEIGHT];
                        }
                    }
                    output[d2][x2][y2] = activationFunction(output[d2][x2][y2]);
                }
            }
        }
        return output;
    }

    @Override
    public void randomize(double min, double max) {
        for (int k = 0; k < increase; k++) {
            for (int i = 0; i < kernelSize; i++) {
                for (int j = 0; j < kernelSize; j++) {
                    weights[k][i][j][WEIGHT] = (min + (max - min) * Math.random());
                    weights[k][i][j][DELTA] = 0;
                }
            }
        }
    }

    @Override
    public double[][][] computeBackwardError(double[][][] input, double[][][] error) {
        if (input == null || !getInputSize().compare(new Size(input)) || error == null || error.length != outputSize.depth)
            throw new IllegalArgumentException();

        double[][][] output = computeOutput(input);
        double[][][] backwardError = new double[inputSize.depth][inputSize.width][inputSize.height];
        for (int d1 = 0; d1 < inputSize.depth; d1++)
            for (int x1 = 0; x1 < inputSize.width; x1++) {
                for (int y1 = 0; y1 < inputSize.height; y1++) {
                    backwardError[d1][x1][y1] = 0;
                    {
                        for (int k = 0; k < increase; k++) {
                            int d2 = d1 * increase + k;
                            for (int x2 = Math.max(0, x1 - kernelSize + 1), i = kernelSize - 1 - x1 + x2; x2 < Math.min(x1, outputSize.width); x2++, i++) {
                                for (int y2 = Math.max(0, y1 - kernelSize + 1), j = kernelSize - 1 - y1 + y2; y2 < Math.min(y1, outputSize.height); y2++, j++) {
                                    backwardError[d1][x1][y1] += error[d2][x2][y2] * weights[k][i][j][WEIGHT] * derivative(output[d2][x2][y2]);
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

        double[][][] output = computeOutput(input);

        for (int d2 = 0; d2 < outputSize.depth; d2++) {
            int d1 = d2 / increase, k = d2 % increase;
            for (int x2 = 0; x2 < outputSize.width; x2++) {
                for (int y2 = 0; y2 < outputSize.height; y2++) {
                    final double gradient = error[d2][x2][y2] * derivative(output[d2][x2][y2]);
                    for (int i = 0, x1 = x2; i < Math.min(inputSize.width - x2, kernelSize); i++, x1++) {
                        for (int j = 0, y1 = y2; j < Math.min(inputSize.height - y2, kernelSize); j++, y1++) {
                            weights[k][i][j][DELTA] = (1. - momentum) * rate * input[d1][x1][y1] * gradient + momentum * weights[k][i][j][DELTA];
                            weights[k][i][j][WEIGHT] += weights[k][i][j][DELTA];
                        }
                    }
                }
            }
        }

    }

    private double activationFunction(double value) {
        return (bipolar ? Math.tanh(value) : 1. / (1. + Math.exp(-value)));
    }

    private double derivative(double value) {
        return (bipolar ? 1. - value * value : value * (1. - value));
    }

}
