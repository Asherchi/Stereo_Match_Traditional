#include <iostream>	
#include <opencv2/opencv.hpp>	
#include <iomanip>
#include "Sad.h"

using namespace std;
using namespace cv;


//constexpr auto invalid = 0xffff;

/** \brief 基础类型别名 */
typedef int8_t			sint8;		// 有符号8位整数
typedef uint8_t			uint8;		// 无符号8位整数
typedef int16_t			sint16;		// 有符号16位整数
typedef uint16_t		uint16;		// 无符号16位整数
typedef int32_t			sint32;		// 有符号32位整数
typedef uint32_t		uint32;		// 无符号32位整数
typedef int64_t			sint64;		// 有符号64位整数
typedef uint64_t		uint64;		// 无符号64位整数
typedef float			float32;	// 单精度浮点
typedef double			float64;	// 双精度浮点

int main()
{

	Mat leftimg = imread("im2.png", 0); // 读取图片
	Mat rightimg = imread("im6.png", 0);
	//Mat leftimg = imread("left8.png", 0);
	//Mat rightimg = imread("right8.png", 0);


	int MaxDisparity = 60;  // 设置视差空间大小， 这个值好像比较大现在
	int winsize = 3;  // 21

	Mat leftPaded, rightPaded;
	if (leftimg.channels() == 3 && rightimg.channels() == 3)  // 转换为灰度图
	{
		cvtColor(leftimg, leftimg, CV_BGR2GRAY);
		cvtColor(rightimg, rightimg, CV_BGR2GRAY);
	}

	int row = leftimg.rows;
	int col = leftimg.cols;

	// 扩充边界 
	copyMakeBorder(leftimg, leftPaded, winsize+1, winsize+1, winsize+1, winsize+1, BORDER_REPLICATE);
	copyMakeBorder(rightimg, rightPaded, winsize + 1, winsize + 1, winsize + 1, winsize + 1, BORDER_REPLICATE);


	std::vector<std::pair<int, int>> occlusions_;
	///** \brief 误匹配区像素集	*/
	std::vector<std::pair<int, int>> mismatches_;

	//resize(leftimg, leftimg, Size(col, row));
	//resize(rightimg, rightimg, Size(col, row));
	Mat depthleft = Mat::zeros(row, col, CV_32S); // 创建内存空间
	Mat depthright = Mat::zeros(row, col, CV_32S);
	Mat lastdisp = Mat::zeros(row, col, CV_32S);
	//Mat left_depthmedian = Mat::zeros(row, col, CV_32S);
	//Mat right_depthmedian = Mat::zeros(row, col, CV_32S);
	//Mat removespeckle = Mat::zeros(row, col, CV_32S);
	//Mat disp_color = Mat::zeros(row, col, CV_32S);


	GetPointDepthLeft(depthleft, leftPaded, rightPaded, MaxDisparity, winsize);  // 左视图
	//GetPointDepthRight(depthright, leftPaded, rightPaded, MaxDisparity, winsize);  // 有视图
	//CrossCheckDiaparity(depthleft, depthright, lastdisp, MaxDisparity, winsize, occlusions_, mismatches_); // 一致性
	//RemoveSpeckles(lastdisp, col, row, 1, 80, Invalid_Float);
	//MatDataNormal(lastdisp, lastdisp);
	//cv::filterSpeckles(lastdisp, Invalid_Float, 100, 1);
	//FillImageNew(lastdisp);
	//FillImageLast(lastdisp);
	//FillImage(lastdisp);
	//FillImageSecondTimes(lastdisp);
	//FillImage(lastdisp);
	//cv::medianBlur(lastdisp, lastdisp, 3); // 必须是奇数窗口
	//MeadianFilter(lastdisp, lastdisp, 2);
	//FillHolesInDispMap(col, row, lastdisp, occlusions_, mismatches_, Invalid_Float);  // 视差填充

	//GetPointDepthLeft(disparity, bytes_left, bytes_right, MaxDisparity, winsize); // removespeckle
	//floodFill(depthleft,)
	cout << "working wiht meadianFilter....." << endl;


	// normal
	MatDataNormal(depthleft, depthleft);
	imshow("left", depthleft);
	imwrite("dispLeft.png", depthleft);
	MatDataNormal(depthright, depthright);
	imwrite("depthright.png", depthright);
	imshow("right", depthright);
	/*MatDataNormal(lastdisp, lastdisp);*/
	imwrite("lastdisp.png", lastdisp);
	imshow("lastDisp", lastdisp);
	waitKey();
	cout << "finished processing ..." << endl;
	return 0;
}