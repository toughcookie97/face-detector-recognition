#include <stdio.h>
#include <algorithm>
#define NOMINMAX
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include<iostream>
#include "mobilefacenet.h"
#include"net.h"
using namespace std;
#include "FaceDetector.h"

using namespace std;

void test_picture()
{

	string imgPath;
	imgPath = "./orignalface.png";
	string param = "./model/face.param";
	string bin = "./model/face.bin";
	const int max_side = 320;

	// slim or RFB
	Detector detector(param, bin, false);
	// retinaface
	// Detector detector(param, bin, true);
	Timer timer;
	for (int i = 0; i < 1; i++) {


		cv::Mat img = cv::imread(imgPath.c_str());

		// scale
		float long_side = std::max(img.cols, img.rows);
		float scale = max_side / long_side;
		cv::Mat img_scale;
		cv::Size size = cv::Size(224, 224);
		cv::resize(img, img_scale, cv::Size(img.cols*scale, img.rows*scale));

		if (img.empty())
		{
			fprintf(stderr, "cv::imread %s failed\n", imgPath.c_str());
			return ;
		}
		std::vector<bbox> boxes;

		timer.tic();

		detector.Detect(img_scale, boxes);
		timer.toc("----total timer:");

		// draw image
		for (int j = 0; j < boxes.size(); ++j) {
			cv::Rect rect(boxes[j].x1 / scale, boxes[j].y1 / scale, boxes[j].x2 / scale - boxes[j].x1 / scale, boxes[j].y2 / scale - boxes[j].y1 / scale);
			cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
			cv::Mat idface = img(rect);
			cv::imwrite("id.png", idface);
			return;
		}
		
	}
	
}

//test_picture***********************************************************************

std::vector<std::string> splitString_1(const std::string &str,
	const char delimiter) {
	std::vector<std::string> splited;
	std::string s(str);
	size_t pos;

	while ((pos = s.find(delimiter)) != std::string::npos) {
		std::string sec = s.substr(0, pos);

		if (!sec.empty()) {
			splited.push_back(s.substr(0, pos));
		}

		s = s.substr(pos + 1);
	}

	splited.push_back(s);

	return splited;
}



float simd_dot_1(const float* x, const float* y, const long& len) {
	float inner_prod = 0.0f;
	__m128 X, Y; // 128-bit values
	__m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
	float temp[4];

	long i;
	for (i = 0; i + 4 < len; i += 4) {
		X = _mm_loadu_ps(x + i); // load chunk of 4 floats
		Y = _mm_loadu_ps(y + i);
		acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
	}
	_mm_storeu_ps(&temp[0], acc); // store acc into an array
	inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

	// add the remaining values
	for (; i < len; ++i) {
		inner_prod += x[i] * y[i];
	}
	return inner_prod;
}
float CalcSimilarity_1(const float* fc1,
	const float* fc2,
	long dim) {

	return simd_dot_1(fc1, fc2, dim)
		/ (sqrt(simd_dot_1(fc1, fc1, dim))
			* sqrt(simd_dot_1(fc2, fc2, dim)));
}

void test_video(string s) {
	
	
	string param = "./model/face.param";
	string bin = "./model/face.bin";
	const int max_side = 320;
	Detector detector(param, bin, false);

	cv::VideoCapture mVideoCapture(0);
	if (!mVideoCapture.isOpened()) {
		return;
	}
	cv::Mat frame;
	mVideoCapture >> frame;
	while (!frame.empty()) {
		mVideoCapture >> frame;
		if (frame.empty()) {
			break;
		}

		clock_t start_time = clock();

		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
		std::vector<bbox> finalBbox;

		detector.Detect(frame, finalBbox);

		const int num_box = finalBbox.size();
		std::vector<cv::Rect> bbox;
		bbox.resize(num_box);
		for (int i = 0; i < num_box; i++) {
			bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);

			
		}
		for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
			rectangle(frame, (*it), cv::Scalar(0, 0, 255), 2, 8, 0);
		}
		imshow("face_detection", frame);
		clock_t finish_time = clock();
		double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
		std::cout << "time" << total_time * 1000 << "ms" << std::endl;
		
		int q = cv::waitKey(10);
		if (q == 27) {
			break;
		}
		else if ((q == 32) && !bbox.empty()) {
			cv::Rect rect(finalBbox[0].x1, finalBbox[0].y1, finalBbox[0].x2  - finalBbox[0].x1 , finalBbox[0].y2  - finalBbox[0].y1 );

			cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
			cv::Mat ROI = frame(rect);
			s += ".png";
			cv::imwrite(s, ROI);
			s = "";
			break;
		}
		


	}
	//return;
}
void compare()
{
	char *model_path = "./model";
	Recognize recognize(model_path);

	cv::Mat img1 = cv::imread("./id.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat img2 = cv::imread("./visitor.png", CV_LOAD_IMAGE_COLOR);
	std::vector<float> feature1;
	std::vector<float> feature2;

	clock_t start_time = clock();
	recognize.start(img1, feature1);
	recognize.start(img2, feature2);
	double similar = calculSimilar(feature1, feature2);
	clock_t finish_time = clock();
	double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;

	std::cout << "time" << total_time * 1000 << "ms" << std::endl;
	std::cout << "similarity is : " << similar << std::endl;
	cv::imshow("left", img1);
	cv::imshow("right", img2);
	cv::waitKey(0);

	
}
int main(int argc, char** argv) {
	
	test_video("id");
	//test_picture();
	test_video("visitor");
	compare();
	
	//test_picture();
	
	return 0;
}