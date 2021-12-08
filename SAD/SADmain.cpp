#include <iostream>	
#include <opencv2/opencv.hpp>	
#include <iomanip>
#include "Sad.h"

using namespace std;
using namespace cv;


//constexpr auto invalid = 0xffff;

/** \brief �������ͱ��� */
typedef int8_t			sint8;		// �з���8λ����
typedef uint8_t			uint8;		// �޷���8λ����
typedef int16_t			sint16;		// �з���16λ����
typedef uint16_t		uint16;		// �޷���16λ����
typedef int32_t			sint32;		// �з���32λ����
typedef uint32_t		uint32;		// �޷���32λ����
typedef int64_t			sint64;		// �з���64λ����
typedef uint64_t		uint64;		// �޷���64λ����
typedef float			float32;	// �����ȸ���
typedef double			float64;	// ˫���ȸ���

int main()
{

	Mat leftimg = imread("im2.png", 0); // ��ȡͼƬ
	Mat rightimg = imread("im6.png", 0);
	//Mat leftimg = imread("left8.png", 0);
	//Mat rightimg = imread("right8.png", 0);


	int MaxDisparity = 60;  // �����Ӳ�ռ��С�� ���ֵ����Ƚϴ�����
	int winsize = 3;  // 21

	Mat leftPaded, rightPaded;
	if (leftimg.channels() == 3 && rightimg.channels() == 3)  // ת��Ϊ�Ҷ�ͼ
	{
		cvtColor(leftimg, leftimg, CV_BGR2GRAY);
		cvtColor(rightimg, rightimg, CV_BGR2GRAY);
	}

	int row = leftimg.rows;
	int col = leftimg.cols;

	// ����߽� 
	copyMakeBorder(leftimg, leftPaded, winsize+1, winsize+1, winsize+1, winsize+1, BORDER_REPLICATE);
	copyMakeBorder(rightimg, rightPaded, winsize + 1, winsize + 1, winsize + 1, winsize + 1, BORDER_REPLICATE);


	std::vector<std::pair<int, int>> occlusions_;
	///** \brief ��ƥ�������ؼ�	*/
	std::vector<std::pair<int, int>> mismatches_;

	//resize(leftimg, leftimg, Size(col, row));
	//resize(rightimg, rightimg, Size(col, row));
	Mat depthleft = Mat::zeros(row, col, CV_32S); // �����ڴ�ռ�
	Mat depthright = Mat::zeros(row, col, CV_32S);
	Mat lastdisp = Mat::zeros(row, col, CV_32S);
	//Mat left_depthmedian = Mat::zeros(row, col, CV_32S);
	//Mat right_depthmedian = Mat::zeros(row, col, CV_32S);
	//Mat removespeckle = Mat::zeros(row, col, CV_32S);
	//Mat disp_color = Mat::zeros(row, col, CV_32S);


	GetPointDepthLeft(depthleft, leftPaded, rightPaded, MaxDisparity, winsize);  // ����ͼ
	//GetPointDepthRight(depthright, leftPaded, rightPaded, MaxDisparity, winsize);  // ����ͼ
	//CrossCheckDiaparity(depthleft, depthright, lastdisp, MaxDisparity, winsize, occlusions_, mismatches_); // һ����
	//RemoveSpeckles(lastdisp, col, row, 1, 80, Invalid_Float);
	//MatDataNormal(lastdisp, lastdisp);
	//cv::filterSpeckles(lastdisp, Invalid_Float, 100, 1);
	//FillImageNew(lastdisp);
	//FillImageLast(lastdisp);
	//FillImage(lastdisp);
	//FillImageSecondTimes(lastdisp);
	//FillImage(lastdisp);
	//cv::medianBlur(lastdisp, lastdisp, 3); // ��������������
	//MeadianFilter(lastdisp, lastdisp, 2);
	//FillHolesInDispMap(col, row, lastdisp, occlusions_, mismatches_, Invalid_Float);  // �Ӳ����

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