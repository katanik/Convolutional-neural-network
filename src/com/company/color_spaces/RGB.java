package com.company.color_spaces;

public class RGB {
    private double R, G, B;

    public RGB(){R=0; G=0; B=0;}

    public RGB(int argb){
        R=(argb >> 16) & 0xff; G=(argb >> 8) & 0xff; B=argb & 0xff;
    }

    public RGB(double r, double g, double b){
        R=r;
        G=g;
        B=b;
    }

    public void setRGB(double r, double g, double b){
        R=r;
        G=g;
        B=b;
    }

    public void setRGB(int argb){
        R=(argb >> 16) & 0xff; G=(argb >> 8) & 0xff; B=argb & 0xff;
    }

    public RGB multiplyDouble(double value){
        return new RGB(R*value, G*value, B*value);
    }

    public RGB square(){
        return new RGB(R*R, G*G, B*B);
    }

    public RGB sqrt(){
        return new RGB(Math.sqrt(R), G=Math.sqrt(G), B=Math.sqrt(B));
    }

    public RGB addRGB(RGB rgb){
        return new RGB(R+rgb.R, G+rgb.G, B+rgb.B);
    }

    public void add(RGB rgb){
        R+=rgb.R; B+=rgb.B; G+=rgb.G;
    }

    public void multiply(double val){
        R*=val; B*=val; G*=val;
    }
    public double getR(){return R;}
    public double getG(){return G;}
    public double getB(){return B;}

    public String out(){
        return "red = " + Double.toString(R) + ",  green = " + Double.toString(G) + ",  blue = " +Double.toString(B);
    }

    public String outInteger() {
        return "red = " + Integer.toString((int)Math.round(R)) +
                ",  green = " + Integer.toString((int)Math.round(G)) +
                ",  blue = " + Integer.toString((int)Math.round(B));
    }

    public int getIntRGB(int alpha){
        return alpha*256*256*256+(int)R*256*256+(int)G*256+(int)B;
    }

}
