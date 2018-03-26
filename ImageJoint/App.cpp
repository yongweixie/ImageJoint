#include <iostream>  
#include <stdio.h>  
#include<opencv2\opencv.hpp>
#include "opencv2/features2d.hpp"  
#include"opencv2/xfeatures2d.hpp"  
#include<Windows.h>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;
Mat sticth(Mat leftimg, Mat rightimg)
{
	Mat leftgray, rightgray;
	cvtColor(leftimg, leftgray, COLOR_BGR2GRAY);
	cvtColor(rightimg, rightgray, COLOR_BGR2GRAY);

	Ptr<SURF> surf;      //创建方式和2中的不一样  
	surf = SURF::create(800);

	BFMatcher matcher;
	Mat leftdes, rightdes;
	vector<KeyPoint>key1, key2;
	vector<DMatch> matches;

	surf->detectAndCompute(leftgray, Mat(), key1, leftdes);
	surf->detectAndCompute(rightgray, Mat(), key2, rightdes);

	matcher.match(leftdes, rightdes, matches);       //匹配  

	sort(matches.begin(), matches.end());  //筛选匹配点  
	vector< DMatch > good_matches;
	int ptsPairs = min(50, (int)(matches.size() * 0.15));
	//cout << ptsPairs << endl;
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]);
	}
	Mat outimg;
	drawMatches(leftimg, key1, rightimg, key2, good_matches, outimg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		obj.push_back(key1[good_matches[i].queryIdx].pt);
		scene.push_back(key2[good_matches[i].trainIdx].pt);
	}

	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0);
	obj_corners[1] = Point(leftimg.cols, 0);
	obj_corners[2] = Point(leftimg.cols, leftimg.rows);
	obj_corners[3] = Point(0, leftimg.rows);
	std::vector<Point2f> scene_corners(4);

	Mat H = findHomography(obj, scene, RANSAC);      //寻找匹配的图像  
	perspectiveTransform(obj_corners, scene_corners, H);
	int start = 0;
	start = leftimg.cols - scene_corners[1].x;

	//line(outimg, scene_corners[0] + Point2f((float)leftimg.cols, 0), scene_corners[1] + Point2f((float)leftimg.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);       //绘制  
	//line(outimg, startPoint1, startPoint2, Scalar(0, 255, 0), 2, LINE_AA);
	//line(outimg, scene_corners[2] + Point2f((float)leftimg.cols, 0), scene_corners[3] + Point2f((float)leftimg.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
	//line(outimg, scene_corners[3] + Point2f((float)leftimg.cols, 0), scene_corners[0] + Point2f((float)leftimg.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
	//imshow("aaaa", outimg);
	//Mat homo = findHomography(key1, key2);
	Mat shftMat = (Mat_<double>(3, 3) << 1.0, 0, start, 0, 1.0, 0, 0, 0, 1.0);
	Mat resultimg;
	warpPerspective(leftimg, resultimg, shftMat*H, Size(leftimg.cols + start, rightimg.rows));
	rightimg.copyTo(Mat(resultimg, Rect(start, 0, rightimg.cols, rightimg.rows)));
	return resultimg;
}
int main()
{
	LARGE_INTEGER t0, t, freq;
	QueryPerformanceCounter(&t0);
	Mat leftimg = imread("1.jpg");     
	Mat rightimg = imread("2.jpg");
	Mat resultimg = sticth(leftimg, rightimg);
	devView(leftimg);
	devView(rightimg);
	devView(resultimg);
	QueryPerformanceCounter(&t);
	QueryPerformanceFrequency(&freq);
	double ms = (t.QuadPart - t0.QuadPart) * 1000 / freq.QuadPart;
	cout << ms << endl;
	cvWaitKey(0);
	system("pause");
}
