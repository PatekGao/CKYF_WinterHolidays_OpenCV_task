#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <experimental/filesystem>
using namespace cv;
using namespace std;

namespace fs= std :: experimental :: filesystem;

bool xcmp(Point a,Point b)
{
    return a.x<b.x;
}
bool ycmp(Point a,Point b)
{
    return a.x<b.x;
}
int x=9,y=12;
vector<vector<int>> pipi(x,vector<int>(y));


vector<Point> juxing(71);
bool XL(int x1,int y1,int x2,int y2){
    if(y1!=y2) return false;
    int minX = min(x1,x2);
    int maxX = max(x1,x2);
    for(int i = minX + 1;i <=maxX - 1;i++){
        if(pipi[i][y1]!=0){
            return false;
        }
    }
    return true;
}

bool YL(int x1,int y1,int x2,int y2){
    if(x1!=x2) return false;
    int minY = min(y1,y2);
    int maxY = max(y1,y2);
    for(int i = minY + 1;i <=maxY - 1;i++){
        if(pipi[x1][i]!=0){
            return false;
        }
    }
    return true;
}

bool ZeroL(int x1,int y1,int x2,int y2){
    if(XL(x1,y1,x2,y2)){
        return true;
    }
    if(YL(x1,y1,x2,y2)){
        return true;
    }
    return false;
}

bool OneL(int x1,int y1,int x2,int y2){
    int tmpPointX[2] = {x1,x2};
    int tmpPointY[2] = {y1,y2};
    for(size_t i = 0; i < 2 ;i++){
        if(ZeroL(tmpPointX[i],tmpPointY[i],x1,y1) &&
           ZeroL(tmpPointX[i],tmpPointY[i],x2,y2)&&
           pipi[tmpPointX[i]][tmpPointY[i]]==0){
            return true;
        }
    }
    return false;
}

bool TwoL(int x1,int y1,int x2,int y2){
    for(size_t e = 0;e < 10; e++) {
        int tmpX1 = x1;
        int tmpY1 = e;
        if (e == y1) continue;
        if (tmpX1 == x2 && tmpY1 == y2) continue;
        int tmpX2 = x2;
        int tmpY2 = tmpY1;
        if(ZeroL(tmpX1,tmpY1,tmpX2,tmpY2)&&
           ZeroL(tmpX1,tmpY1,x1,y1)&&
           ZeroL(tmpX2,tmpY2,x2,y2)&&
           pipi[tmpX1][tmpY1]==0&&
           pipi[tmpX2][tmpY2]==0) return true;
    }
    for(size_t e = 0;e < 7; e++){
        int tmpX1 = e;
        int tmpY1 = y1;
        if(e == x1) continue;;
        if(tmpY1 == y2 && tmpX1 == x2) continue;
        int tmpX2 = tmpX1;
        int tmpY2 = y2;
        if(ZeroL(tmpX1,tmpY1,tmpX2,tmpY2)&&
           ZeroL(tmpX1,tmpY1,x1,y1)&&
           ZeroL(tmpX2,tmpY2,x2,y2)&&
           pipi[tmpX1][tmpY1]==0&&
           pipi[tmpX2][tmpY2]==0) return true;
    }
    return false;
}


int main() {
    Mat image=imread("../example.jpg");
    Mat xianshi=imread("../example.jpg");
    resize(image,image,Size(),0.5,0.5);
    imshow("image",image);
    waitKey(0);

    vector<Mat> templates;
    vector<string> names=
            {
                    "../templates/chongya.png",
                    "../templates/dage.png",
                    "../templates/duiyou.png",
                    "../templates/feibiao.png",
                    "../templates/iloverm.png",
                    "../templates/shanghai.png",
                    "../templates/shuangqiang.png",
            };

    for(auto& p : names){
        cout<<"read : " << p << endl;
        templates.push_back(imread(p));
    }



    int minSize;
    double minValRes = -1;
    for(int size = 40;size < 80; size++){
        Mat result;
        //Mat temp = templates[0];
        Mat temp;
        //resize(image,temp,Size(size,size));
        resize(templates[0],temp,Size(size,size));
        int result_cols = image.cols-temp.cols+1;
        int result_rows = image.rows-temp.rows+1;
        result.create(result_rows,result_cols,CV_32FC1);
        matchTemplate(image,temp,result,TM_SQDIFF_NORMED);
        //imshow("result",result);
        //waitKey(0);
        double minVal; double maxVal; Point minLoc; Point maxLoc;
        minMaxLoc(result,&minVal,&maxVal,&minLoc,&maxLoc);
        if(minValRes == -1 || minValRes > minVal){
            minSize = size;
            minValRes = minVal;
        }
    }
    cout << "minSize : " << minSize << endl;
    for(auto &temp: templates){
        resize(temp,temp,Size(minSize,minSize));
    }

    //pipei
    vector<Mat> results;
    for(auto &temp :templates) {
        cout << "matching" << endl;
        Mat result;
        int result_cols = image.cols - temp.cols + 1;
        int result_rows = image.rows - temp.rows + 1;
        result.create(result_rows, result_cols, CV_32FC1);
        matchTemplate(image, temp, result, TM_SQDIFF_NORMED);
        results.push_back(result);
        //  imshow("result",result);
        // waitKey(0);
    }

    //NMS
    int flag=1;
    int  result_cols=image.cols - minSize+1;
    int  result_rows=image.rows - minSize +1;
    Mat final_result(result_cols,result_rows,CV_32FC1);
    vector<vector<Point>> final_loc(0,vector<Point>(0));
    vector<Scalar> mycolors=
            {
                    Scalar (50,10,255),
                    Scalar (10,255,60),
                    Scalar (255,40,10),
                    Scalar (10,200,255),
                    Scalar (150,10,255),
                    Scalar (100,200,100),
                    Scalar (100,180,255),

            };
    int n=0;
    for(auto& result:results)
    {
        Mat result_copy = result.clone();
        vector<Point> true_loc;
        double thres;
        minMaxLoc(result,&thres);
        threshold(result,result,thres*4,0,THRESH_TOZERO_INV);
        vector<Point> result_loc;
        vector<Point> result_loc_tmp;
        findNonZero(result,result_loc_tmp);
        bool dup;
        dup= false;
        for(auto &pt:result_loc_tmp)
        {
            dup= false;
            for(auto& p:result_loc)
            {
                if(abs(p.x-pt.x) <= minSize/1.5 && abs(p.y-pt.y)<=minSize/1.5)
                {
                    dup= true;
                    break;
                }
            }
            if(!dup)
            {
                result_loc.push_back(pt);
                circle(image,pt,5,mycolors[n],2);
            }
        }


        final_loc.push_back(result_loc);
        n++;
    }
    //imshow("result",image);
    //waitKey(0);
    int x_r,y_b,x_l,y_t;
    x_r=y_b=0;
    x_l=y_t=1000;
    for(auto & floc_frame:final_loc)
    {
        x_r = max_element(floc_frame.begin(),floc_frame.end(),xcmp)-> x > x_r ? max_element(floc_frame.begin(),floc_frame.end(),xcmp) -> x : x_r;
        y_b = max_element(floc_frame.begin(),floc_frame.end(),ycmp)-> y > y_b ? max_element(floc_frame.begin(),floc_frame.end(),ycmp) -> y : y_b;
        x_l = min_element(floc_frame.begin(),floc_frame.end(),xcmp)-> x < x_l ? min_element(floc_frame.begin(),floc_frame.end(),xcmp) -> x : x_l;
        y_t = min_element(floc_frame.begin(),floc_frame.end(),ycmp)-> y < y_t ? min_element(floc_frame.begin(),floc_frame.end(),ycmp) -> y : y_t;

    }

    int xnum,ynum;
    xnum=(int)round((x_r-x_l)/(minSize*1.135))+1;
    ynum=(int)round((y_b-y_t)/(minSize*1.135))+1+1;

    cout<<"xnum:"<<xnum<<"ynum:"<<ynum<<endl;

    vector<vector<int>> final_res(ynum,vector<int>(xnum,0));
    int id = 1;
    cout<<"show the result"<<endl;
    for(auto & floc_frame:final_loc)
    {
        for(auto& floc : floc_frame)
        {
            final_res[round((floc.y-y_t)/(minSize*1.135))][round((floc.x-x_l)/(minSize*1.135))]=id;
        }
        id++;

    }

    int ccn;
    for(flag=0;flag<=69;flag++){
        juxing[flag].x=29+51.5*(flag%7);
        ccn=flag/7;
        juxing[flag].y=226+52*ccn;
    }
    for(int i=0;i<=8;i++){
        for(int j=0;j<=11;j++){
            //flag++;
            //cout<< flag <<endl;
            pipi[i][j]=0;
        }
    }
    int i=1;
    int j=1;
    for(auto& final_res_row:final_res){
        for(auto& final_res_col:final_res_row){
            cout<<final_res_col<<" ";
            pipi[i][j]=final_res_col;
            i++;
        }
        cout<<endl;
        i=1;
        j++;
    }
    int rr=0;
    resize(xianshi,xianshi,Size(),0.5,0.5);
    int i1;
    int j1;
    imshow("xianshi",xianshi);
    waitKey(1000);
    for(j=1;j<=10;j++){
        for(i=1;i<=7;i++){
            for(j1=1;j1<=10;j1++){
                for(i1=1;i1<=7;i1++){
                    if(pipi[i][j]==0||pipi[i1][j1]==0) continue;
                    if(i==i1&&j==j1) continue;
                    if(ZeroL(i,j,i1,j1)||OneL(i,j,i1,j1)||TwoL(i,j,i1,j1)){
                        if(pipi[i][j]==pipi[i1][j1]) {
                            pipi[i][j] = 0;
                            pipi[i1][j1] = 0;
                            cout << "(" << j-1 << "," << i-1 << ")" << " " <<"and"<<" "<< "(" << j1-1 << "," << i1-1 << ")" << endl;
                            int tt,ss;
                            tt=(j-1)*7+i-1;
                            ss=(j1-1)*7+i1-1;
                            juxing[tt].x+=25;
                            juxing[tt].y+=25;
                            juxing[ss].x+=25;
                            juxing[ss].y+=25;
                            circle(xianshi,juxing[tt],10,(200,100,0),9);
                            circle(xianshi,juxing[ss],10,(200,100,0),9);
                            imshow("show",xianshi);
                            waitKey(500);
                        }
                    }
                }
            }
        }
    }

}
