#define DEBUG_
#ifdef _WIN32
	const int FPS = 32;
#endif

#include <iostream>
#include <fstream>
#include <stdint.h>
#include <vector>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

//#include "GenericList.hpp"

using namespace cv;
using namespace std;
//using namespace gn;

/* FUNCTION DECL */
void swapQuadrants(Mat& source);
void runDFT(Mat& source, Mat& destination);
void displayDFT(const Mat source);
void filter(const Mat source, Mat& destination);

/*******/
int main(int argv, char** argc)
{
	VideoCapture cam(0);
	if (!cam.isOpened()) { return -1; }

	//const int MAX_IMAGES = 128;
	//##TODO
	//initialize 2 lists
	//one for storing original images
	//one for storing pointers to images with high correlation

#ifdef DEBUG_
	namedWindow("Source_DEBUG", WINDOW_NORMAL);
	//namedWindow("DFT_DEBUG", WINDOW_NORMAL); resizeable window for FourierTransform not necessary
	namedWindow("Result_DEBUG", WINDOW_NORMAL);
#else
	namedWindow("Result", WINDOW_NORMAL);
	setWindowProperty("Result", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
#endif

	bool run = true;
	while (run) {
		Mat frame;
		cam.read(frame);

		if (frame.empty()) { break; }
		
		Mat grayScaleFrame;
		cvtColor(frame, grayScaleFrame, COLOR_BGR2GRAY);

#ifdef DEBUG_
		// draw random pixels to act as noise
		grayScaleFrame.at<char>(10, 100) = 128;
		grayScaleFrame.at<char>(10, 105) = 128;
		grayScaleFrame.at<char>(10, 110) = 128;
		grayScaleFrame.at<char>(10, 115) = 128;
		grayScaleFrame.at<char>(100, 100) = 128;
		grayScaleFrame.at<char>(105, 100) = 128;
		grayScaleFrame.at<char>(110, 100) = 128;
		grayScaleFrame.at<char>(115, 100) = 128;

		imshow("Source_DEBUG", grayScaleFrame);
#endif

		Mat srcFloat;
		grayScaleFrame.convertTo(srcFloat, CV_32F, 1.0 / 255.0);

		Mat dftResult;
		runDFT(srcFloat, dftResult);

#ifdef DEBUG_
		displayDFT(dftResult);
#endif

		Mat filtered;
		filter(dftResult, filtered);

		// use inverse flags in DFT function to invert Fourier Transform to original image
		Mat invertedDFT;
		dft(filtered, invertedDFT, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

		//##TODO
		//store filtered images in list
		//add up and take average image -> reference image
		//find correlation of reference image and ORIGINAL of other images
		//throw away images in list with low correlation
		//**program continues to collect images in place of these
		//take average image of ORIGINALS of images with high correlation with offset applied
		//these images stay until their lifespan is over

#ifdef DEBUG_
		imshow("Result_DEBUG", invertedDFT);
#else
		imshow("Result", invertedDFT);
#endif

#ifdef _WIN32
		if (waitKey(1000 / FPS) >= 0) { break; }
#endif
	}
	
	return 0;
}

/*
 * Rearrange the quadrants in an image matrix.
 *
 * @param [in] source Source matrix of which the quadrants are swapped.
*/
void swapQuadrants(Mat& source) {
	int cutX = source.cols / 2;
	int cutY = source.rows / 2;

	Mat topLeft(source, Rect(0, 0, cutX, cutY));
	Mat topRight(source, Rect(cutX, 0, cutX, cutY));
	Mat bottomLeft(source, Rect(0, cutY, cutX, cutY));
	Mat bottomRight(source, Rect(cutX, cutY, cutX, cutY));

	Mat temp;
	
	// top left with bottom right
	topLeft.copyTo(temp);
	bottomRight.copyTo(topLeft);
	temp.copyTo(bottomRight);

	// top right with bottom left
	topRight.copyTo(temp);
	bottomLeft.copyTo(topRight);
	temp.copyTo(bottomLeft);
}

/*
 * Takes the Fourier Transform of an image.
 * 
 * @param [in] source Source image
 * @param [in] result Resulting Mat object stored here
*/
void runDFT(Mat& source, Mat& result) 
{
	// 2 channels in complex for real and imaginary component
	Mat srcComplex[2] = { source, Mat::zeros(source.size(), source.type()) };

	// merge into new DFT ready matrix object with 2 channels
	Mat dftReady;
	merge(srcComplex, 2, dftReady);

	// take DFT and store in result
	dft(dftReady, result, DFT_COMPLEX_OUTPUT);
}

/*
 * Displays the proper Fourier Transform of an image.
 *
 * @param [in] source The DFT source to be displayed
*/
void displayDFT(const Mat source) 
{
	// split the source
	Mat splitArray[2] = { Mat::zeros(source.size(), source.type()), Mat::zeros(source.size(), source.type()) };
	split(source, splitArray);


	// swap quadrants
	swapQuadrants(splitArray[0]);
	swapQuadrants(splitArray[1]);

	// define radii of the circles for the frequency domain
	const int INNER_RADIUS = 0.02;
	const int OUTER_RADIUS = 36000;

	int x, y;

	for (int row = 0; row < source.rows; row++) {
		for (int col = 0; col < source.cols; col++) {
			x = col - source.cols / 2;
			y = row - source.rows / 2;

			// remove frequencies - inner circle
			//if (((x * x) + (y * y)) <= INNER_RADIUS) {
			//	splitArray[0].at<float>(row, col) = splitArray[0].at<float>(row, col) * 0.0;
			//	splitArray[1].at<float>(row, col) = splitArray[1].at<float>(row, col) * 0.0;
			//}

			// remove frequencies - outer circle
			if (((x * x) + (y * y)) >= OUTER_RADIUS) {
				splitArray[0].at<float>(row, col) = splitArray[0].at<float>(row, col) * 0.0;
				splitArray[1].at<float>(row, col) = splitArray[1].at<float>(row, col) * 0.0;
			}
		}
	}

	imshow("split_1_DEBUG_UNALTERED", splitArray[0]);

	// revert quadrants
	swapQuadrants(splitArray[0]);
	swapQuadrants(splitArray[1]);

	imshow("split_1_DEBUG", splitArray[0]);
	imshow("split_2_DEBUG", splitArray[1]);
	
	// take magnitude
	Mat dftMagnitude;
	magnitude(splitArray[0], splitArray[1], dftMagnitude);

	// add 1 to all values and take the log of the values
	dftMagnitude += Scalar::all(1);
	log(dftMagnitude, dftMagnitude);

	// normalize matrix and display in a window
	normalize(dftMagnitude, dftMagnitude, 0, 1, NORM_MINMAX);
	imshow("DFT_DEBUG", dftMagnitude);
	//waitKey();
}

/*
 * Splits the 2 DFT channels into its imaginary and real components in individual matrices,
 * then swaps the quadrants to rearrange the frequencies,
 * then removes unneeded frequencies,
 * lastly, rearranges the quadrants back to their initial position for inverting back to an image.
 * 
 * @param [in] source The source image of which the frequencies will be changed
 * @param [in] destination The destination in which the changed matrix will be stored in.
*/
void filter(const Mat source, Mat& destination)
{
	Mat splitArray[2] = { Mat::zeros(source.size(), source.type()), Mat::zeros(source.size(), source.type()) };
	split(source, splitArray);

	swapQuadrants(splitArray[0]);
	swapQuadrants(splitArray[1]);

	const float INNER_RADIUS = 0.01;
	const float OUTER_RADIUS = 36000;
	
	int x, y;

	for (int row = 0; row < source.rows; row++) {
		for (int col = 0; col < source.cols; col++) {
			// center of image
			x = col - source.cols / 2;
			y = row - source.rows / 2;
			// formula of a circle: x^2 + Y^2 = r^2

			// remove frequencies - inner circle
			//if (((x * x) + (y * y)) < INNER_RADIUS) {
			//	splitArray[0].at<float>(row, col) = splitArray[0].at<float>(row, col) * 0.5;
			//	splitArray[1].at<float>(row, col) = splitArray[1].at<float>(row, col) * 0.5;
			//}

			// remove frequencies - outer circle
			if (((x * x) + (y * y)) >= OUTER_RADIUS) {
				splitArray[0].at<float>(row, col) = splitArray[0].at<float>(row, col) * 0.0;
				splitArray[1].at<float>(row, col) = splitArray[1].at<float>(row, col) * 0.0;
			}
		}
	}

	swapQuadrants(splitArray[0]);
	swapQuadrants(splitArray[1]);

	merge(splitArray, 2, destination);
}
