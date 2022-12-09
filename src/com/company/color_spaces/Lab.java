package com.company.color_spaces;

import static java.lang.Math.*;

public class Lab {
    private double L, a, b;
    private double X, Y, Z;
    private double X1, Y1, Z1;
    private double Xn=95.047, Yn=100.0, Zn=108.883;

    public Lab(){}

    public void setLab(double L, double a, double b){
        this.L=L; this.a=a; this.b=b;
    }

    public RGB toRGB(){
        double Y1=(L+16.)/116.;
        double X1=a/500.+Y1;
        double Z1=Y1-b/200.;
        if (pow(Y1, 3.)>0.008856) Y1= pow(Y1, 3.);
        else Y1=(Y1-16./116.)/7.787;
        if (pow(X1, 3.)>0.008856) X1= pow(X1, 3.);
        else X1=(X1-16./116.)/7.787;
        if (pow(Z1, 3.)>0.008856) Z1= pow(Z1, 3.);
        else Z1=(Z1-16./116.)/7.787;
        X=X1*Xn/100.;
        Y=Y1*Yn/100.;
        Z=Z1*Zn/100.;

        double R1=X*3.2406+Y*(-1.5372)+Z*(-0.4986);
        double G1=X*(-0.9689)+Y*1.8758+Z*0.0415;
        double B1=X*0.0557+Y1*(-0.2040)+Z*1.0570;

        if (R1>0.0031308) R1=1.055*(pow(R1, 1./2.4))-0.055;
        else R1=12.92*R1;
        if (G1>0.0031308) G1=1.055*(pow(G1, 1./2.4))-0.055;
        else G1=12.92*G1;
        if (B1>0.0031308) B1=1.055*(pow(B1, 1./2.4))-0.055;
        else B1=12.92*B1;

        return new RGB(R1*255., G1*255., B1*255.);

    }

    public Lab(RGB rgb){
        this.setLab(rgb);
    }

    public double CIEDE2000(Lab lab){
        double L1=L, a1=a, b1=b, L2=lab.L, a2=lab.a, b2=lab.b;
        double C1ab=sqrt(a1*a1+b1*b1);
        double C2ab=sqrt(a2*a2+b2*b2);
        double C_ab=(C1ab+C2ab)/2.;
        double G=0.5*(1-sqrt(pow(C_ab, 7.)/(pow(C_ab, 7.)+pow(25., 7.))));
        double a_1=(1+G)*a1, a_2=(1+G)*a2;
        double C1=sqrt(pow(a_1, 2.)+pow(b1, 2.));
        double C2=sqrt(pow(a_2, 2.)+pow(b2, 2.));
        double h1, h2;
        if (b1==0 && a_1==b1)
            h1=0.;
        else{
            h1=atan2(b1, a_1);
            if (h1<0.)
                h1+=360.*(PI/180.);
        }

        if (b2==0 && a_2==b2)
            h2=0.;
        else{
            h2=atan2(b2, a_2);
            if (h2<0.)
                h2+=360.*(PI/180.);
        }

        double deltaL=L2-L1;
        double deltaC=C2-C1;
        double deltah;
        if (C1*C2==0.) deltah=0.;
        else{
            if (abs(h2-h1)<=180.) {
                deltah=h2-h1;
            }
            else{
                if (h2-h1>180.)
                    deltah=h2-h1-360.;
                else
                    deltah=h2-h1+360.;
            }
        }
        double deltaH=2.*sqrt(C1*C2)*sin(deltah/2.);
        double L=(L1+L2)/2.;
        double C=(C1+C2)/2.;

        double h;

        if (C1*C2==0){
            h=h1+h2;
        }
        else {
            if (abs(h2 - h1) <= 180.) {
                h = (h1 + h2) / 2.;
            }
            else{
                if (h1+h2<360.)
                    h=(h1+h2+360.)/2.;
                else
                    h=(h1+h2-360.)/2.;
            }
        }

        double T=1-0.17*cos(h-30.)+0.24*cos(2*h)+0.32*cos(3*h+6)-0.20*cos(4*h-63);
        double deltaPhi=30*exp(-((h-275.)/25.));
        double RC=2.*sqrt(Math.pow(C, 7.)/(Math.pow(C, 7.)+pow(25., 7.)));
        double SL=1+(0.015*Math.pow(L-50., 2.)/sqrt(20.+Math.pow(L-50., 2.)));
        double SC=1.+0.045*C;
        double SH=1+0.015*C*T;
        double RT=-sin(2*deltaPhi)*RC;

        double kL = 1.0, kC = 1.0, kH = 1.0;
        double deltaE = sqrt(
                Math.pow(deltaL/(kL*SL), 2.)+
                        Math.pow(deltaC/(kC*SC), 2.)+
                        Math.pow(deltaH/(kH*SH), 2.)+
                        RT*(deltaC/(kC*SC))*(deltaH*(kH*SH))
        );

        return deltaE;
    }

    public void setLab(RGB rgb) {
        double R = rgb.getR() / 255., G = rgb.getG() / 255., B = rgb.getB() / 255.;

        if (R > 0.04045) R = pow((R + 0.055) / 1.055, 2.4);
        else R /= 12.92;
        if (G > 0.04045) G = pow((G + 0.055) / 1.055, 2.4);
        else G /= 12.92;
        if (B > 0.04045) B = pow((B + 0.055) / 1.055, 2.4);
        else B /= 12.92;

        R *= 100.;
        G *= 100.;
        B *= 100.;

        X1=X = 0.4124 * R + 0.3576 * G + 0.1805 * B;
        Y1=Y = 0.2126 * R + 0.7152 * G + 0.0722 * B;
        Z1=Z = 0.0193 * R + 0.1192 * G + 0.9505 * B;

        X /= Xn;
        Y /= Yn;
        Z /= Zn;

        if (X > 0.008856) X = pow(X, 1. / 3.);
        else X = (7.787 * X) + (16. / 116.);
        if (Y > 0.008856) Y = pow(Y, 1. / 3.);
        else Y = (7.787 * Y) + (16. / 116.);
        if (Z > 0.008856) Z = pow(Z, 1. / 3.);
        else Z = (7.787 * Z) + (16. / 116.);

        L = 116. * Y - 16.;
        a = 500. * (X - Y);
        b = 200. * (Y - Z);
    }

    public String out() {
        return "L* = " + Double.toString(L) + ",  a* = " + Double.toString(a) + ",  b* = " + Double.toString(b);
    }

    public String outInteger() {
        return "L* = " + Integer.toString((int) Math.round(L)) +
                ",  a* = " + Integer.toString((int) Math.round(a)) +
                ",  b* = " + Integer.toString((int) Math.round(b));
    }

    public double getL(){return L;}
    public double geta(){return a;}
    public double getb(){return b;}

}
