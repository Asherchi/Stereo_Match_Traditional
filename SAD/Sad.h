#pragma once
#include <iostream>	
#include <opencv2/opencv.hpp>	
#include <iomanip>
#include <vector>	

using namespace std;
using namespace cv;

#define UINT8_MAX  0xffui8
#define UINT16_MAX  0xffffui16
//constexpr auto invalid = 0xffui8;
constexpr auto Invalid_Float = std::numeric_limits<float>::infinity();

float sadvalue(const Mat& src1, const Mat& src2)
{
	Mat  matdiff = cv::abs(src1 - src2);
	int  saddiff = cv::sum(matdiff)[0];
	return saddiff;
}

float GetMinSadIndex(std::vector<float>& sad)
{
	float minsad = sad[0];
	int index = 0;
	int len = sad.size();
#pragma omp parallel for  
	for (int i = 1; i < len; ++i)
	{
		if (sad[i] < minsad)
		{
			minsad = sad[i];
			index = i;
		}
	}

	return index;
}

float OptimalDisparity(std::vector<float>& sad, const int disp_range)
{
	float minDisp_value = 0xffff;
	float best_disp = 0xffff;
	float secMin = sad[0];
#pragma omp parallel for  
	for (auto i = 1; i < sad.size(); ++i)
	{
		if (minDisp_value > sad[i])
		{
			minDisp_value = sad[i];
			best_disp = i;
		}
	}

	for (auto i = 0; i < sad.size(); ++i)
	{
		if (minDisp_value == sad[i])
			continue;
		else
		{
			const auto cost = sad[i];
			secMin = min(secMin, cost);
		}
	}

	if (secMin - minDisp_value <= 0.01)
	{
		return 0;
	}

	if (best_disp == 0 || best_disp == disp_range - 1)
	{
		return 0;
	}

	const int index1 = best_disp - 1;
	const int index2 = best_disp + 1;
	const float cost1 = sad[index1];
	const float cost2 = sad[index2];
	const float divided = max(static_cast<float>(1), cost1 + cost2 - 2 * minDisp_value);
	const float final_disp = best_disp + (cost1 - cost2) / (2 * divided);
	//cout << " best disp is:" << best_disp << endl;

	return best_disp;
}

void MatDataNormal(const Mat& src, Mat& dst)
{
	// ��һ����Ŀ�ļ����֮����ʹ��û�пɱ��Ե����ݱ�þ��пɱ��ԣ�ͬʱ�ֱ�����Ƚϵ���������֮�����Թ�ϵ
	normalize(src, dst, 255, 0, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);
}



void GetPointDepthLeft(Mat& disparity, const  Mat& leftimg, const Mat& rightimg,
	const int MaxDisparity, const  int winsize)
{
	int row = leftimg.rows;
	int col = leftimg.cols;
	if (leftimg.channels() == 3 && rightimg.channels() == 3)
	{
		cvtColor(leftimg, leftimg, CV_BGR2GRAY);
		cvtColor(rightimg, rightimg, CV_BGR2GRAY);
	}

	//Mat disparity = Mat ::zeros(row,col, CV_32S);
	//int w = winsize-1;
	int win_r = winsize+1;   // ����Ϊʲô��1��һ������ͼ��������
	int rowrange = row - winsize-1;
	int colrange = col - winsize-1;
#pragma omp parallel for  
	for (int i = win_r; i < rowrange; ++i)
	{
		int* ptr = disparity.ptr<int>(i- win_r);
		for (int j = win_r; j < colrange; ++j)  // ��������˴�����Ӳ�λ�ÿ�ʼ���ᵼ��ͼ���С
		{
			//Rect leftrect;
			Mat leftwin = leftimg(Range(i-win_r, i + win_r + 1), Range(j-win_r, j + win_r + 1));
			std::vector<float> sad(MaxDisparity);
			for (int d = 0; d < MaxDisparity; ++d)  // ��ȡ���ҵ��Ӳ�ķ�����ȫ��һ��
			{  
				// ����Ǵ���ͼ���Ҳ�ֵ��ʼ�Ƚϣ�Ȼ��������һ��һ���Ƚ�
				//Rect rightrect;
				if (j-win_r - d < 0)
				{
					sad[d] = sad[d-1];
					continue;
				}
				Mat rightwin = rightimg(Range(i-win_r, i + win_r + 1), Range(j-win_r-d, j-d+ win_r + 1));
				sad[d] = sadvalue(leftwin, rightwin);
			}
			//*(ptr + j) = GetMinSadIndex(sad);
			*(ptr + j- win_r) = OptimalDisparity(sad, MaxDisparity);
		}
		double rate = double(i) / (colrange);  // showing the process
		cout << "Depth left finished" << setprecision(2) << rate * 100 << "%" << endl;
	}
}

void GetPointDepthRight(Mat& disparity, const Mat& leftimg, const Mat& rightimg,
	const int MaxDisparity, const  int winsize)
{
	int row = leftimg.rows;
	int col = leftimg.cols;
	if (leftimg.channels() == 3 && rightimg.channels() == 3)
	{
		cvtColor(leftimg, leftimg, CV_BGR2GRAY);
		cvtColor(rightimg, rightimg, CV_BGR2GRAY);
	}

	//Mat disparity = Mat ::zeros(row,col, CV_32S);
	int w = winsize+1;
	int rowrange = row - w-1;
	int colrange = col - w-1;
#pragma omp parallel for  
	for (int i = w; i < rowrange; ++i)
	{
		int* ptr = disparity.ptr<int>(i-w);
		for (int j = w; j < colrange; ++j)
		{
			//Rect rightrect;
			Mat rightwin = rightimg(Range(i - w, i + w + 1), Range(j - w, j + w + 1));
			std::vector<float> sad(MaxDisparity);
			for (int d = 0; d < MaxDisparity; ++d)
			{
				if (j + d + w + 1 > col)
				{
					sad[d] = sad[d-1];
					continue;
				}
				//Rect leftrect;
				Mat leftwin = leftimg(Range(i - w, i + w + 1), Range(j - w+d , j +d+ w + 1));
				sad[d] = sadvalue(leftwin, rightwin);   
			}   

			*(ptr + j- w) = GetMinSadIndex(sad);
		}
		double rate = double(i) / (colrange);  // showing the process
		cout << "Depth right finished" << setprecision(2) << rate * 100 << "%" << endl;
	}
}

void CrossCheckDiaparity(const Mat& leftdisp, const Mat& rightdisp, Mat& lastdisp,
	const int MaxDisparity, const int winsize, vector<pair<int,int>> &occlusion, vector<pair<int,int>> &mismatch)
{
	int row = leftdisp.rows;
	int col = leftdisp.cols;
	int w = winsize;
	int rowrange = row;  
	int colrange = col;
	int diffthreshold = 5;
	occlusion.clear();
	mismatch.clear();
//#pragma omp parallel for  
	for (int i = 0; i < row; ++i)
	{
		const int* ptrleft = leftdisp.ptr<int>(i);  // ���������ĳһ�е��Ӳ�ֵ
		const int* ptrright = rightdisp.ptr<int>(i);
		int* ptrdisp = lastdisp.ptr<int>(i);
		for (int j = 0; j < col; ++j)
		{
			int leftvalue = *(ptrleft + j);  // ������������������ʲô����һ�����꣬�൱��ȡ��ĳһ�еĵ�i�е�����
			int rightvalue = *(ptrright + j - leftvalue);  // ΪʲôҪ��ȥ���Ӳ�ֵ�أ���Ϊ��ͼƥ����ͼ��ʱ���ҵ������ص�λ�øպò���Ӳ�ֵ������ͼ��������
			int diff = abs(leftvalue - rightvalue);  //�õ���Ӧ�Ӳ�ֵ֮�󣬾Ϳ�����һ���Լ�����
			if (diff > diffthreshold)
			{
				if (leftvalue < rightvalue)
					occlusion.emplace_back(i, j);
				else
					mismatch.emplace_back(i, j);
				*(ptrdisp + j) = Invalid_Float;
			}
			else
			{
				*(ptrdisp + j) = leftvalue;
			}
		}
		//cout << "the process is" << static_cast<float>(i / row)*100 << "%" << endl;
	}

}

void MeadianFilter(const Mat& disp_in,  Mat& disp_out, const int  winSize)
{
	int half_size = winSize / 2;
	int rol = disp_in.rows;
	int col = disp_in.cols;
	vector<int> number;
	number.reserve(rol * col);
	for (int i = 0; i < rol; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			number.clear();
			for (int r = -half_size; r <= half_size; r++) {
				for (int c = -half_size; c <= half_size; c++) {
					const int rowin = i + r;
					const int colin = j + c;
					if (rowin >= 0 && rowin < rol && colin >= 0 && colin < col) {
						number.push_back(disp_in.at<int>(rowin,colin));
					}
				}
			}
			sort(number.begin(), number.end());
			disp_out.at<uchar>(i, j) = uchar(number[number.size() / 2]);
		}
	}
}

void RemoveSpeckles(Mat disparity_map, const int& width, const int& height,
	const int& diff_insame, const int& min_speckle_aera, const float& invalid_val)
{
	assert(width > 0 && height > 0);
	if (width < 0 || height < 0) {
		//return -1;
		cout << " errors happen !" << endl;
	}

	// �����������Ƿ���ʵ�����
	std::vector<bool> visited(int(width * height), false);
#pragma omp parallel for  
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (visited[i * width + j] || disparity_map.at<float>(i,j) == 0) {
				// �����ѷ��ʵ����ؼ���Ч����
				continue;
			}
			// ������ȱ������������
			// ����ͨ�����С����ֵ�������Ӳ�ȫ��Ϊ��Чֵ
			std::vector<std::pair<int,int>> vec;
			vec.emplace_back(i, j);
			visited[i * width + j] = true;
			int cur = 0;
			int next = 0;
			do {
				// ������ȱ����������	
				next = vec.size();
				for (int k = cur; k < next; k++) {
					const auto& pixel = vec[k];
					const int row = pixel.first;
					const int col = pixel.second;
					const auto& disp_base = disparity_map.at<float>(row,col);
					// 8�������
					for (int r = -1; r <= 1; r++) {
						for (int c = -1; c <= 1; c++) {
							if (r == 0 && c == 0) {
								continue;
							}
							int rowr = row + r;
							int colc = col + c;
							if (rowr >= 0 && rowr < height && colc >= 0 && colc < width) {
								if (!visited[rowr * width + colc] &&
									(disparity_map.at<float>(rowr,colc) != invalid_val) &&
									abs(disparity_map.at<float>(rowr,colc) - disp_base) <= diff_insame) {
									vec.emplace_back(rowr, colc);
									visited[rowr * width + colc] = true;
								}
							}
						}
					}
				}
				cur = next;
			} while (next < vec.size());

			// ����ͨ�����С����ֵ�������Ӳ�ȫ��Ϊ��Чֵ
			if (vec.size() < min_speckle_aera) {
				for (auto& pix : vec) {
					//disparity_map[pix.first * width + pix.second] = invalid_val;
					disparity_map.at<float>(pix.first, pix.second) = invalid_val;
				}
			}
		}
	}
}

void FillHolesInDispMap(const int width_, const int height_, const Mat& disp_left_, vector<pair<int,int>> occlusion, vector<pair<int,int>> mismatch,
	const float Invalid_Float)
{
	const int width = width_;
	const int height = height_;

	std::vector<float> disp_collects;

	// ����8������
	float pi = 3.1415926;
	float angle1[8] = { pi, 3 * pi / 4, pi / 2, pi / 4, 0, 7 * pi / 4, 3 * pi / 2, 5 * pi / 4 };
	float angle2[8] = { pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4 };
	float* angle = angle1;

	Mat disp_ptr = disp_left_;
#pragma omp parallel for  
	for (int k = 0; k < 3; k++) {
		// ��һ��ѭ�������ڵ������ڶ���ѭ��������ƥ����
		auto& trg_pixels = (k == 0) ? occlusion : mismatch;

		std::vector<std::pair<int, int>> inv_pixels;
		if (k == 2) {
			//  ������ѭ������ǰ����û�д���ɾ�������
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if (disp_ptr.at<float>(i,j) == Invalid_Float) {
						inv_pixels.emplace_back(i, j);
					}
				}
			}
			trg_pixels = inv_pixels;
		}

		// ��������������
		for (auto& pix : trg_pixels) {
			int y = pix.first;
			int x = pix.second;

			if (y == height / 2) {
				angle = angle2;
			}

			// �ռ�8���������������׸���Ч�Ӳ�ֵ
			disp_collects.clear();
			for (int n = 0; n < 8; n++) {
				const float ang = angle[n];
				const float sina = sin(ang);
				const float cosa = cos(ang);
				for (int n = 1; ; n++) {
					const int yy = y + n * sina;
					const int xx = x + n * cosa;
					if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
						break;
					}
					//auto& disp = *(disp_ptr + yy * width + xx);
					auto& disp = disp_ptr.at<float>(yy,xx);
					if (disp != Invalid_Float) {
						disp_collects.push_back(disp);
						break;
					}
				}
			}
			if (disp_collects.empty()) {
				continue;
			}

			std::sort(disp_collects.begin(), disp_collects.end());

			// ������ڵ�������ѡ��ڶ�С���Ӳ�ֵ
			// �������ƥ��������ѡ����ֵ
			if (k == 0) {
				if (disp_collects.size() > 1) {
					disp_ptr.at<float>(y,x) = disp_collects[1];
				}
				else {
					disp_ptr.at<float>(y, x) = disp_collects[0];
				}
			}
			else {
				disp_ptr.at<float>(y, x) = disp_collects[disp_collects.size() / 2];
			}
		}
	}
}


void FillImage(Mat& disp)
{
	int col = disp.cols;
	int row = disp.rows;
	vector<pair<int, int>> Nonvalue;
	vector<int> dispValue;

	for (auto i = 0; i < row; i++)
	{
		for (auto j = 0; j < col; j++)
		{
			auto value = disp.at<uchar>(i, j);
			if (value)
			{
				continue;
			}
			else
			{
				Nonvalue.emplace_back(i, j);
			}
		}
	}
	for (auto i = 0; i < Nonvalue.size(); i++)
	{
		auto pixel = Nonvalue[i];
		int pixel_row = pixel.first;
		int pixel_col = pixel.second;
		int offset = 0;
		if(pixel_col >= 0)
		{
			while (pixel_col >= 0)
			{
				int x = pixel_row;
				pixel_col = pixel_col - offset;
				if (pixel_col < 0)
				{
					dispValue.push_back(0);
					break;
				}
				auto value = disp.at<uchar>(x, pixel_col);
				if (value)
				{
					dispValue.push_back(value);
					break;
				}
				++offset;
			}
		}
		else 
		{
			while (pixel_col < col)
			{
				int x = pixel_row;
				pixel_col = pixel_col + offset;
				if (pixel_col > col)
				{
					dispValue.push_back(0);
					break;
				}
				auto value = disp.at<uchar>(x, pixel_col);
				if (value)
				{
					dispValue.push_back(value);
					break;
				}
				++offset;
			}
		}
	
	}
	for (auto i = 0; i <Nonvalue.size(); i++)
	{
		auto pixelN = Nonvalue[i];
		int pixel_row = pixelN.first;
		int pixel_col = pixelN.second;
		disp.at<uchar>(pixel_row, pixel_col) = dispValue[i];
	}
}


void FillImageSecondTimes(Mat& disp)
{
	int col = disp.cols;
	int row = disp.rows;
	vector<pair<int, int>> Nonvalue;
	vector<int> dispValue;

	for (auto i = 0; i < row; i++)
	{
		for (auto j = 0; j < col; j++)
		{
			auto value = disp.at<int>(i, j);
			if (value>20)
			{
				continue;
			}
			else
			{
				Nonvalue.emplace_back(i, j);
			}
		}
	}
	for (auto i = 0; i < Nonvalue.size(); i++)
	{
		auto pixel = Nonvalue[i];
		int pixel_row = pixel.first;
		int pixel_col = pixel.second;
		int offset = 0;
		while (pixel_col < col)
		{
			int x = pixel_row;
			pixel_col = pixel_col + offset;
			if (pixel_col > col)
			{
				dispValue.push_back(uchar(20));
				break;
			}
			auto value = disp.at<uchar>(x, pixel_col);
			if (value)
			{
				dispValue.push_back(value);
				break;
			}
			++offset;
		}
		

	}
	for (auto i = 0; i < Nonvalue.size(); i++)
	{
		auto pixelN = Nonvalue[i];
		int pixel_row = pixelN.first;
		int pixel_col = pixelN.second;
		disp.at<uchar>(pixel_row, pixel_col) = dispValue[i];
	}
}


void FillImageNew(Mat& disp)
{
	int col = disp.cols;
	int row = disp.rows;
	vector<pair<int, int>> Nonvalue;
	vector<int> dispValue;

	for (auto i = 0; i < row; i++)
	{
		for (auto j = 0; j < col; j++)
		{
			auto value = disp.at<uchar>(i, j);
			if (value)
			{
				continue;
			}
			else
			{
				Nonvalue.emplace_back(i, j);
			}
		}
	}
	for (auto i = 0; i < Nonvalue.size(); i++)
	{
		auto pixel = Nonvalue[i];
		int pixel_row = pixel.first;
		int pixel_col = pixel.second;
		int offset = 0;
		
		if (pixel_col >= 0)
		{
			while (pixel_col >= 0)
			{
				int x = pixel_row;
				pixel_col = pixel_col - offset;
				if (pixel_col < 0)
				{
					pixel_col = pixel.second;  // �����趨��ʼֵ
					offset = 0;
					//dispValue.push_back(0);
					break;
				}
				auto value = disp.at<uchar>(x, pixel_col);
				if (value)
				{
					dispValue.push_back(value);
					pixel_col = 0xffff;  // Ϊ��ʹ�����if��ִ��
					break;
				}
				++offset;
			}
	         while (pixel_col < col)
			{
				int x = pixel_row;
				pixel_col = pixel_col + offset;
				if (pixel_col > col)
				{
					dispValue.push_back(0);
					break;
				}
				auto value = disp.at<uchar>(x, pixel_col);
				if (value)
				{
					dispValue.push_back(value);
					break;
				}
				++offset;
			}
		}
	}
	for (auto i = 0; i < Nonvalue.size(); i++)
	{
		auto pixelN = Nonvalue[i];
		int pixel_row = pixelN.first;
		int pixel_col = pixelN.second;
		disp.at<uchar>(pixel_row, pixel_col) = dispValue[i];
	}
}


void FillImageLast(Mat& disp)
{
	int col = disp.cols;
	int row = disp.rows;
	vector<pair<int, int>> Nonvalue;
	vector<int> dispValue;

	for (auto i = 0; i < row; i++)
	{
		for (auto j = 0; j < col; j++)
		{
			auto value = disp.at<uchar>(i, j);
			if (value)
			{
				continue;
			}
			else
			{
				Nonvalue.emplace_back(i, j);
			}
		}
	}
	for (auto i = 0; i < Nonvalue.size(); i++)
	{
		auto pixel = Nonvalue[i];
		int pixel_row = pixel.first;
		int pixel_col = pixel.second;
		int offset = 0;

		if (pixel_row >= 0)
		{
			while (pixel_row >= 0)
			{
				int y = pixel_col;
				pixel_row = pixel_row - offset;
				if (pixel_row < 0)
				{
					pixel_row = pixel.first;  // �����趨��ʼֵ
					offset = 0;
					//dispValue.push_back(0);
					break;
				}
				auto value = disp.at<uchar>(pixel_row,y);
				if (value)
				{
					dispValue.push_back(value);
					pixel_row = 0xffff;  // Ϊ��ʹ�����if��ִ��
					break;
				}
				++offset;
			}
			while (pixel_row < row)
			{
				int y = pixel_col;
				pixel_row = pixel_row + offset;
				if (pixel_row > row)
				{
					dispValue.push_back(0);
					break;
				}
				auto value = disp.at<uchar>(pixel_row, y);
				if (value)
				{
					dispValue.push_back(value);
					break;
				}
				++offset;
			}
		}
	}
	for (auto i = 0; i < Nonvalue.size(); i++)
	{
		auto pixelN = Nonvalue[i];
		int pixel_row = pixelN.first;
		int pixel_col = pixelN.second;
		disp.at<uchar>(pixel_row, pixel_col) = dispValue[i];
	}
}
