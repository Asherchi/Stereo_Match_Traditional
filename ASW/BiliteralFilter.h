#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

////////////////////////////
//获取高斯模板（空间模板）
///////////////////////////
void getGausssianMask(cv::Mat& Mask, cv::Size wsize, double spaceSigma = 10) {
	Mask.create(wsize, CV_64F);
	int h = wsize.height;
	int w = wsize.width;
	int center_h = (h - 1) / 2;
	int center_w = (w - 1) / 2;
	double sum = 0.0;
	double x, y;

	for (int i = 0; i < h; ++i) {
		y = pow(i - center_h, 2);
		double* Maskdate = Mask.ptr<double>(i);
		for (int j = 0; j < w; ++j) {
			x = pow(j - center_w, 2);
			double g = exp(-(x + y) / (2 * spaceSigma * spaceSigma));
			Maskdate[j] = g;
			sum += g;
		}
	}
}

////////////////////////////
//获取色彩模板（值域模板）
///////////////////////////
void getColorMask(std::vector<double>& colorMask, double colorSigma= 30) {

	for (int i = 0; i < 256; ++i) {
		double colordiff = exp(-(i * i) / (2 * colorSigma * colorSigma));
		colorMask.push_back(colordiff);
	}

}


////////////////////////////
//双边滤波
///////////////////////////
void bilateralfiter(cv::Mat& src, cv::Mat& dst, cv::Size wsize, double spaceSigma, double colorSigma) {
	cv::Mat spaceMask;
	std::vector<double> colorMask;
	cv::Mat Mask0 = cv::Mat::zeros(wsize, CV_64F);
	cv::Mat Mask1 = cv::Mat::zeros(wsize, CV_64F);
	cv::Mat Mask2 = cv::Mat::zeros(wsize, CV_64F);

	getGausssianMask(spaceMask, wsize, spaceSigma);//空间模板
	getColorMask(colorMask, colorSigma);//值域模板
	int hh = (wsize.height - 1) / 2;
	int ww = (wsize.width - 1) / 2;
	dst.create(src.size(), src.type());
	//边界填充
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, ww, ww, cv::BORDER_REPLICATE);//边界复制;

	for (int i = hh; i < src.rows + hh; ++i) {
		for (int j = ww; j < src.cols + ww; ++j) {
			double sum[3] = { 0 };   // 为啥这里是三不是其他的啊 噢 三个通道
			int graydiff[3] = { 0 };
			double space_color_sum[3] = { 0.0 };

			for (int r = -hh; r <= hh; ++r) {
				for (int c = -ww; c <= ww; ++c) {
					if (src.channels() == 1) {
						int centerPix = Newsrc.at<uchar>(i, j);
						int pix = Newsrc.at<uchar>(i + r, j + c);
						graydiff[0] = abs(pix - centerPix);
						double colorWeight = colorMask[graydiff[0]];
						Mask0.at<double>(r + hh, c + ww) = colorWeight * spaceMask.at<double>(r + hh, c + ww);//滤波模板 
						space_color_sum[0] = space_color_sum[0] + Mask0.at<double>(r + hh, c + ww);

					}
					else if (src.channels() == 3) {
						cv::Vec3b centerPix = Newsrc.at<cv::Vec3b>(i, j);
						cv::Vec3b bgr = Newsrc.at<cv::Vec3b>(i + r, j + c);
						graydiff[0] = abs(bgr[0] - centerPix[0]); graydiff[1] = abs(bgr[1] - centerPix[1]); graydiff[2] = abs(bgr[2] - centerPix[2]);
						double colorWeight0 = colorMask[graydiff[0]];
						double colorWeight1 = colorMask[graydiff[1]];
						double colorWeight2 = colorMask[graydiff[2]];
						Mask0.at<double>(r + hh, c + ww) = colorWeight0 * spaceMask.at<double>(r + hh, c + ww);//滤波模板
						Mask1.at<double>(r + hh, c + ww) = colorWeight1 * spaceMask.at<double>(r + hh, c + ww);
						Mask2.at<double>(r + hh, c + ww) = colorWeight2 * spaceMask.at<double>(r + hh, c + ww);
						space_color_sum[0] = space_color_sum[0] + Mask0.at<double>(r + hh, c + ww);
						space_color_sum[1] = space_color_sum[1] + Mask1.at<double>(r + hh, c + ww);
						space_color_sum[2] = space_color_sum[2] + Mask2.at<double>(r + hh, c + ww);
					}
				}
			}

			//滤波模板归一化
			if (src.channels() == 1)
				Mask0 = Mask0 / space_color_sum[0];
			else {
				Mask0 = Mask0 / space_color_sum[0];
				Mask1 = Mask1 / space_color_sum[1];
				Mask2 = Mask2 / space_color_sum[2];
			}


			for (int r = -hh; r <= hh; ++r) {
				for (int c = -ww; c <= ww; ++c) {

					if (src.channels() == 1) {
						sum[0] = sum[0] + Newsrc.at<uchar>(i + r, j + c) * Mask0.at<double>(r + hh, c + ww); //滤波
					}
					else if (src.channels() == 3) {
						cv::Vec3b bgr = Newsrc.at<cv::Vec3b>(i + r, j + c); //滤波
						sum[0] = sum[0] + bgr[0] * Mask0.at<double>(r + hh, c + ww);//B
						sum[1] = sum[1] + bgr[1] * Mask1.at<double>(r + hh, c + ww);//G
						sum[2] = sum[2] + bgr[2] * Mask2.at<double>(r + hh, c + ww);//R
					}
				}
			}

			for (int k = 0; k < src.channels(); ++k) {
				if (sum[k] < 0)
					sum[k] = 0;
				else if (sum[k] > 255)
					sum[k] = 255;
			}
			if (src.channels() == 1)
			{
				dst.at<uchar>(i - hh, j - ww) = static_cast<uchar>(sum[0]);
			}
			else if (src.channels() == 3)
			{
				cv::Vec3b bgr = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				dst.at<cv::Vec3b>(i - hh, j - ww) = bgr;
			}
		}
	}

}


void bilateralfiterWight(cv::Mat& src, cv::Mat& dst, int wsize,const Mat& spaceMask, vector<double> colorMask) 
{
	cv::Mat spaceMask;
	std::vector<double> colorMask;
	cv::Mat Mask0 = cv::Mat::zeros(src.rows, src.cols, CV_64F);
	cv::Mat Mask1 = cv::Mat::zeros(src.rows, src.cols, CV_64F);
	cv::Mat Mask2 = cv::Mat::zeros(src.rows, src.cols, CV_64F);

	//getGausssianMask(spaceMask, cv::Size(wsize*2+3), spaceSigma);//空间模板
	//getColorMask(colorMask, colorSigma);//值域模板
	//int hh = (wsize.height - 1) / 2;
	//int ww = (wsize.width - 1) / 2;
	int center_x = wsize + 1;  // wsize 是半径
	int center_y = wsize + 1;
	dst.create(src.size(), src.type());
	//边界填充
	//cv::Mat Newsrc;
	//cv::copyMakeBorder(src, Newsrc, hh, hh, ww, ww, cv::BORDER_REPLICATE);//边界复制;
	double sum[3] = { 0 };   // 为啥这里是三不是其他的啊 噢 三个通道
	int graydiff[3] = { 0 };
	double space_color_sum[3] = { 0.0 };

	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			int centerPix = src.at<uchar>(center_x, center_y);
			//for (int r = -hh; r <= hh; ++r) {
			//	for (int c = -ww; c <= ww; ++c) {
			if (src.channels() == 1) {
				int pix = src.at<uchar>(i, j);
				graydiff[0] = abs(pix - centerPix);
				double colorWeight = colorMask[graydiff[0]];
				Mask0.at<double>(i, j) = colorWeight * spaceMask.at<double>(i, j);//滤波模板 
				space_color_sum[0] = space_color_sum[0] + Mask0.at<double>(i, j);

			}
			else if (src.channels() == 3) {
				cv::Vec3b centerPix = src.at<cv::Vec3b>(i, j);
				cv::Vec3b bgr = src.at<cv::Vec3b>(i, j);
				graydiff[0] = abs(bgr[0] - centerPix[0]); graydiff[1] = abs(bgr[1] - centerPix[1]); graydiff[2] = abs(bgr[2] - centerPix[2]);
				double colorWeight0 = colorMask[graydiff[0]];
				double colorWeight1 = colorMask[graydiff[1]];
				double colorWeight2 = colorMask[graydiff[2]];
				Mask0.at<double>(i, j) = colorWeight0 * spaceMask.at<double>(i, j);//滤波模板
				Mask1.at<double>(i, j) = colorWeight1 * spaceMask.at<double>(i, j);
				Mask2.at<double>(i, j) = colorWeight2 * spaceMask.at<double>(i, j);
				space_color_sum[0] = space_color_sum[0] + Mask0.at<double>(i, j);
				space_color_sum[1] = space_color_sum[1] + Mask1.at<double>(i, j);
				space_color_sum[2] = space_color_sum[2] + Mask2.at<double>(i, j);
			}
		}
	}

	//滤波模板归一化
	if (src.channels() == 1)
		Mask0 = Mask0 / space_color_sum[0];
	else {
		Mask0 = Mask0 / space_color_sum[0];
		Mask1 = Mask1 / space_color_sum[1];
		Mask2 = Mask2 / space_color_sum[2];
	}


	for (int i = 0; i < src.rows; ++i)
	{
		double pixel;
		for (int j = 0; j < src.cols; ++j) 
		{
			if (src.channels() == 1)
			{
				pixel = src.at<uchar>(i, j ) * Mask0.at<double>(i, j); //滤波
			}
			else if (src.channels() == 3) 
			{
				cv::Vec3b bgr = src.at<cv::Vec3b>(i, j); //滤波
				sum[0] = sum[0] + bgr[0] * Mask0.at<double>(i, j);//B
				sum[1] = sum[1] + bgr[1] * Mask1.at<double>(i, j);//G
				sum[2] = sum[2] + bgr[2] * Mask2.at<double>(i, j);//R
			}

			for (int k = 0; k < src.channels(); ++k) 
			{
				if (sum[k] < 0)
					sum[k] = 0;
				else if (sum[k] > 255)
					sum[k] = 255;
			}
			if (src.channels() == 1)
			{
				dst.at<uchar>(i, j) = static_cast<uchar>(pixel);
			}
			else if (src.channels() == 3)
			{
				cv::Vec3b bgr = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				dst.at<cv::Vec3b>(i, j) = bgr;
			}
		}
	}
}

//int main() {
//	//cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\woman1.jpeg");
//	cv::Mat src = cv::imread("im2.png");
//	if (src.empty()) {
//		return -1;
//	}
//	cv::Mat dst;
//	double spaceSigma = 10;
//	double colorSigma = 30;
//	bilateralfiter(src, dst, cv::Size(23, 23), spaceSigma, colorSigma);
//	cv::namedWindow("src", CV_WINDOW_NORMAL);
//	cv::imshow("src", src);
//	cv::namedWindow("双边滤波", CV_WINDOW_NORMAL);
//	cv::imshow("双边滤波", dst);
//	cv::imwrite("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Image Filtering\\BilateralFilter\\woman1.jpg", dst);
//	cv::waitKey(0);
//	return 0;
//}