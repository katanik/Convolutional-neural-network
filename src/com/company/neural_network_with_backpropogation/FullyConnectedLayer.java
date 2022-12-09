package com.company.neural_network_with_backpropogation;

import com.company.neural_network.Size;

public final class FullyConnectedLayer implements BackpropogationLayer {
    private double[][][][][][][] weights;
    private Size inputSize, outputSize;
    /**
     * If bipolar is true then activation function is hyperbolic tangent
     * If bipolar is false then activation function is sigmoid function
     */
    private final boolean bipolar;

    private final int WEIGHT = 0;
    private final int DELTA = 1;

    public FullyConnectedLayer(Size inputSize, Size outputSize, boolean bipolar) {
        if (inputSize == new Size(0, 0, 0))
            throw new IllegalArgumentException();

        this.inputSize = inputSize;
        this.outputSize = outputSize;
        weights = new double[inputSize.depth][inputSize.width][inputSize.height][outputSize.depth][outputSize.width][outputSize.height][2];
        this.bipolar = bipolar;
    }

    public FullyConnectedLayer(Size inputSize, Size outputSize) {
        this(inputSize, outputSize, false);
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
            for (int x2 = 0; x2 < outputSize.width; x2++) {
                for (int y2 = 0; y2 < outputSize.height; y2++) {
                    for (int d1 = 0; d1 < inputSize.depth; d1++) {
                        for (int x1 = 0; x1 < inputSize.width; x1++) {
                            for (int y1 = 0; y1 < inputSize.height; y1++) {
                                output[d2][x2][y2] += weights[d1][x1][y1][d2][x2][y2][WEIGHT] * input[d1][x1][y1];
                            }
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
        for (int d1 = 0; d1 < inputSize.depth; d1++) {
            for (int x1 = 0; x1 < inputSize.width; x1++) {
                for (int y1 = 0; y1 < inputSize.height; y1++) {
                    for (int d2 = 0; d2 < outputSize.depth; d2++) {
                        for (int x2 = 0; x2 < outputSize.width; x2++) {
                            for (int y2 = 0; y2 < outputSize.height; y2++) {
                                weights[d1][x1][y1][d2][x2][y2][WEIGHT] = min + (max - min) * Math.random();
                                weights[d1][x1][y1][d2][x2][y2][DELTA] = 0.;
                            }
                        }
                    }
                }
            }
        }
    }

    @Override
    public double[][][] computeBackwardError(double[][][] input, double[][][] error) {
        if (input == null || !inputSize.compare(new Size(input)) || error == null || !outputSize.compare(new Size(error)))
            throw new IllegalArgumentException();

        double[][][] output = computeOutput(input);
        double[][][] backwardError = new double[inputSize.depth][inputSize.width][inputSize.height];

        for (int d1 = 0; d1 < inputSize.depth; d1++) {
            for (int x1 = 0; x1 < inputSize.width; x1++) {
                for (int y1 = 0; y1 < inputSize.height; y1++) {
                    backwardError[d1][x1][y1] = 0;
                    for (int d2 = 0; d2 < outputSize.depth; d2++) {
                        for (int x2 = 0; x2 < outputSize.width; x2++) {
                            for (int y2 = 0; y2 < outputSize.height; y2++) {
                                backwardError[d1][x1][y1] += error[d2][x2][y2] * weights[d1][x1][y1][d2][x2][y2][WEIGHT] * derivative(output[d2][x2][y2]);
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
        if (input == null || !inputSize.compare(new Size(input)) || error == null || !outputSize.compare(new Size(error)))
            throw new IllegalArgumentException();

        double[][][] output = computeOutput(input);

        for (int d2 = 0; d2 < outputSize.depth; d2++) {
            for (int x2 = 0; x2 < outputSize.width; x2++) {
                for (int y2 = 0; y2 < outputSize.height; y2++) {
                    final double gradient = error[d2][x2][y2] * derivative(output[d2][x2][y2]);
                    for (int d1 = 0; d1 < inputSize.depth; d1++) {
                        for (int x1 = 0; x1 < inputSize.width; x1++) {
                            for (int y1 = 0; y1 < inputSize.height; y1++) {
                                weights[d1][x1][y1][d2][x2][y2][DELTA] = (1. - momentum) * rate * input[d1][x1][y1] * gradient + momentum * weights[d1][x1][y1][d2][x2][y2][DELTA];
                                weights[d1][x1][y1][d2][x2][y2][WEIGHT] += weights[d1][x1][y1][d2][x2][y2][DELTA];
                            }
                        }
                    }
                }
            }
        }
    }

    private double activationFunction(double value) {
        return (bipolar == true ? Math.tanh(value) : 1. / (1. + Math.exp(-value)));
    }

    private double derivative(double value) {
        return (bipolar ? 1. - value * value : value * (1. - value));
    }
}
