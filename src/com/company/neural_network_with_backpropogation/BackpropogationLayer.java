package com.company.neural_network_with_backpropogation;

import com.company.neural_network.Layer;

/**
 * Interface layer trained by backpropogation error
 */
public interface BackpropogationLayer extends Layer {

    /**
     * Gives random values to the weights of neurons
     * @param min minimum value
     * @param max maximum value
     */
    void randomize(double min, double max);

    /**
     * Compute next error vector backward
     * @param input input vector
     * @param error error vector
     * @return next error vector backward
     */
    double[][][] computeBackwardError(double[][][] input, double[][][] error);

    /**
     * Adjust the weights of neurons in direction of reducing the error
     * @param input input vector
     * @param error error vector
     * @param rate training rate
     * @param momentum momentum
     */
    void adjust (double[][][] input, double[][][] error, double rate, double momentum);

}
