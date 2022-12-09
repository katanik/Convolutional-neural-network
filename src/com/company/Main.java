package com.company;

import com.company.color_spaces.RGB;
import com.company.filters.GaborFilter;
import com.company.neural_network.Layer;
import com.company.neural_network.Network;
import com.company.neural_network.Size;
import com.company.neural_network_with_backpropogation.BackpropogationNetwork;
import com.company.neural_network_with_backpropogation.ConvolutionalLayer;
import com.company.neural_network_with_backpropogation.FullyConnectedLayer;
import com.company.neural_network_with_backpropogation.PoolingLayer;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;

public class Main {
    private static int convolutionKernel = 5, poolingKernel = 3, increaseLayer = 2;
    private static double rate = 0.01, momentum = 0.001;
    private static int classNumber = 3, layersNumber = 3;

    private static Size sizeImage = new Size(3, 48, 48);
    private static int numberOfTrainingData = 45, numberOfTestData = 30;
    private static double[][][][] input, output;

    private static String nameClasses[] = {"tree", "cloud", "swan"};

    public static Layer[] buildingNetwork() {
        Layer[] layers = new Layer[layersNumber];
        Size size = sizeImage;
        for (int i = 0; i < layersNumber - 1; i += 2) {
            layers[i] = new ConvolutionalLayer(size, convolutionKernel, increaseLayer, false);
            layers[i + 1] = new PoolingLayer(layers[i].getSize(), poolingKernel);
            size = layers[i + 1].getSize();
        }

        FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(
                layers[layersNumber - 2].getSize(),
                new Size(classNumber, 1, 1), false);
        layers[layersNumber - 1] = fullyConnectedLayer;
        return layers;
    }

    public static double[][][] loadImage(String path) {
        double pattern[][][] = null;
        try {
            BufferedImage bufferedImage = ImageIO.read(new File(path));
            int width = bufferedImage.getWidth(), height = bufferedImage.getHeight();

            pattern = new double[3][width][height];
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    int argb = bufferedImage.getRGB(x, y);
                    pattern[0][x][y] = ((argb >> 16) & 0xff);
                    pattern[1][x][y] = ((argb >> 8) & 0xff);
                    pattern[2][x][y] = (argb & 0xff);
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        pattern = GaborFilter.applyKernal(sizeImage.width, sizeImage.height, pattern);

        return pattern;
    }

    public static void addPattern(int index, double[][][] pattern, int rightAnswer) {
        input[index] = pattern;
        double[][][] goal = new double[classNumber][1][1];
        goal[rightAnswer][0][0] = 1.;
        output[index] = goal;
    }

    public static void printNetworkAnswer(String path, double[][][] answer) {
        if (answer.length != classNumber)
            throw new IllegalArgumentException("Depth of answer don't equal number of classes.");
        JFrame window = new JFrame("Image from file \"" + path+"\"");
        window.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        window.setBounds(200, 200, 450, 350);
        JPanel panel = new JPanel();
        panel.setBackground(Color.WHITE);
        panel.setLayout(null);
        panel.setSize(450, 350);
        JLabel image = new JLabel(new ImageIcon(path));
        image.setBackground(Color.WHITE);
        image.setBounds(0, 0, 450, 200);
        panel.add(image, BorderLayout.CENTER);
        String text = "<html>";

        double sum = 0;
        long cur, rest = 100;
        for (int i = 0; i < classNumber; i++) {
            if (answer[i] == null) throw new IllegalArgumentException("Answer has null cell.");
            sum += answer[i][0][0];
        }

        /*for (int i = 0; i < classNumber; i++) {
            text += nameClasses[i] + " = " + Double.toString(answer[i][0][0]) + "<br>";
        }*/

        for (int i = 0; i < classNumber - 1; i++) {
            cur = Math.round(answer[i][0][0] * 100. / sum);
            rest -= cur;
            text += nameClasses[i] + " = " + Long.toString(cur) + " %   <br>";
        }
        text += nameClasses[classNumber - 1] + " = " + Long.toString(rest) + " %   <br>";

        text += "</html>";
        JLabel message = new JLabel(text, SwingConstants.CENTER);
        message.setBounds(0, 200, 450, 150);
        message.setBackground(Color.WHITE);
        panel.add(message);
        window.add(panel);
        window.setVisible(true);
    }

    public static void loadImagesForLearning() {
        //buildingData(1, 15, "training_data");

        input = new double[numberOfTrainingData][][][];
        output = new double[numberOfTrainingData][][][];
        for (int i = 0; i < numberOfTrainingData; i++) {
            try (BufferedReader in = new BufferedReader(new FileReader(getPathClass(i + 1, "training_data")))) {
                int nameOfClass = in.read() - '0';
                addPattern(i, loadImage(getPathImage(i + 1, "training_data")), nameOfClass);
            } catch (IOException | NumberFormatException e) {
                e.printStackTrace();
            }
        }
    }

    public static void learning() {
        Layer[] layers = buildingNetwork();
        BackpropogationNetwork network = new BackpropogationNetwork(layers);
        int iteration = 0;
        int epoch = 500;
        double error = 10;
        while (error > 0.2) {
            iteration++;
            error = 0.;
            for (int i = 0; i < numberOfTrainingData; i++) {
                error = Math.max(error, network.learnPattern(input[i], output[i], rate, momentum));
            }
        }
        System.out.print(iteration);
        network.saveToFile("neural_network");
    }

    public static void printMetrics() {
        BackpropogationNetwork network = (BackpropogationNetwork) Network.loadFromFile("neural_network");
        int TP[] = new int[classNumber];
        int TN[] = new int[classNumber];
        int FP[] = new int[classNumber];
        int FN[] = new int[classNumber];

        int numberOfTrueAnswer = 0;
        for (int i = 1; i <= numberOfTestData; i++) {
            try (BufferedReader in = new BufferedReader(new FileReader(getPathClass(i, "training_data")))) {
                int nameOfClass = in.read() - '0';
                int index = -1;
                double maxAnswer = -1.;
                double answer[][][] = network.computeOutput(loadImage(getPathImage(i, "test_data")));
                for (int j = 0; j < classNumber; j++) {
                    if (maxAnswer < answer[j][0][0]) {
                        maxAnswer = answer[j][0][0];
                        index = j;
                    }
                }
                if (index == nameOfClass) {
                    numberOfTrueAnswer++;
                    for (int j = 0; j < classNumber; j++) {
                        if (j == nameOfClass)
                            TP[j]++;
                        else
                            TN[j]++;
                    }
                } else {
                    for (int j = 0; j < classNumber; j++) {
                        if (j == nameOfClass)
                            FN[j]++;
                        else if (j == index)
                            FP[j]++;
                        else
                            TN[j]++;
                    }
                }

            } catch (IOException | NumberFormatException e) {
                e.printStackTrace();
            }
        }

        JFrame messageWindow = new JFrame("Metrics");
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setSize(350, 300);
        messageWindow.setBounds(700, 200, 350, 350);
        panel.setBackground(Color.WHITE);
        panel.add(new JLabel("Total accuracy = " + Double.toString((double) numberOfTrueAnswer / (double) numberOfTestData)));
        for (int j = 0; j < classNumber; j++) {
            panel.add(new JLabel(" "));
            panel.add(new JLabel("Recall for class \"" + nameClasses[j] + "\" = " + Double.toString((double) TP[j] / (double) (TP[j] + FN[j]))));
            panel.add(new JLabel("Precision for class \"" + nameClasses[j] + "\" = " + Double.toString((double) TP[j] / (double) (TP[j] + FP[j]))));
            panel.add(new JLabel("Accuracy for class \"" + nameClasses[j] + "\" = " + Double.toString((double) (TP[j] + TN[j]) / numberOfTestData)));
        }
        messageWindow.add(panel);
        messageWindow.setVisible(true);
    }

    public static void printSomeResults() {
        BackpropogationNetwork network = (BackpropogationNetwork) Network.loadFromFile("neural_network");
        String path;

        for (int i = 1; i <= 10; i++) {
            path = getPathImage(i, "test_data");
            printNetworkAnswer(path, network.computeOutput(loadImage(path)));
        }
    }

    public static void main(String[] args) {
        //buildingData(16, 25, "test_data");
        //GaborFilter.buildKernel();
        //loadImagesForLearning();
        //learning();
        printSomeResults();
        printMetrics();
    }


    public static String getPathImage(int index, String type) {
        return ".\\src\\" + type + "\\images\\" + Integer.toString(index) + ".jpg";
    }

    public static String getPathClass(int index, String type) {
        return ".\\src\\" + type + "\\classes\\" + Integer.toString(index) + ".txt";
    }

    public static void checkFilter(String path, String pathForSaving) {
        try {
            BufferedImage bImage = ImageIO.read(new File(path));
            double pattern[][][] = new double[3][bImage.getWidth()][bImage.getHeight()];

            for (int i = 0; i < bImage.getWidth(); i++) {
                for (int j = 0; j < bImage.getHeight(); j++) {
                    int argb = bImage.getRGB(i, j);
                    pattern[0][i][j] = (argb >> 16) & 0xff;
                    pattern[1][i][j] = (argb >> 8) & 0xff;
                    pattern[2][i][j] = argb & 0xff;
                }
            }

            pattern = GaborFilter.applyKernal(bImage.getWidth(), bImage.getHeight(), pattern);

            for (int i = 0; i < bImage.getWidth(); i++) {
                for (int j = 0; j < bImage.getHeight(); j++) {
                    bImage.setRGB(i, j, new RGB(pattern[0][i][j], pattern[1][i][j], pattern[2][i][j]).getIntRGB(0));
                }
            }

            try {
                ImageIO.write(bImage, "jpg", new File(pathForSaving));
            } catch (IOException e) {
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void buildingData(int indexBegin, int indexEnd, String type) {
        String path;
        int key = 1, nameOfClass = 0;

        for (int i = indexBegin; i <= indexEnd; i++) {
            for (int j = 0; j < 3; j++, key++) {
                path = ".\\src\\patterns\\labels\\";
                switch (j) {
                    case 0:
                        path += "trees\\";
                        nameOfClass = 0;
                        break;
                    case 1:
                        path += "clouds\\";
                        nameOfClass = 1;
                        break;
                    default:
                        path += "swans\\";
                        nameOfClass = 2;
                        break;
                }
                path += Integer.toString(i) + ".jpg";
                try {
                    ImageIO.write(ImageIO.read(new File(path)), "jpg", new File(getPathImage(key, type)));
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
                try (FileWriter fileWithClass = new FileWriter(new File(getPathClass(key, type)), false)) {
                    fileWithClass.write(Integer.toString(nameOfClass));
                    fileWithClass.flush();
                } catch (IOException e2) {
                    e2.printStackTrace();
                }
            }
        }

    }
}
