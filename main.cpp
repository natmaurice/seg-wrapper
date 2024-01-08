#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include <algorithm>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <random>
#include <cmath>

#define OPTPARSE_IMPLEMENTATION
#include <optparse.h>



enum class Algorithm {
    THRESHOLD,
    OTSU,
    CCL,
    WATERSHED,
    KMEANS,
    UNKNOWN,
};

enum class OutputStyle {
    NONE,
    BORDER,
    COLOR,
};

void color_regions(cv::Mat& input_mat, cv::Mat& output_mat, int label_count);

std::string uppercase(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(),
		   [](unsigned char c) {
		       return std::toupper(c);
		   });
    return str;
}

Algorithm get_algorithm_from_name(std::string name) {    

    name = uppercase(name);
    
    if (name == "THRESHOLD") {
	return Algorithm::THRESHOLD;
    } else if (name == "OTSU") {
	return Algorithm::OTSU;
    } else if (name == "CCL") {
	return Algorithm::CCL;
    } else if (name == "WATERSHED") {
	return Algorithm::WATERSHED;
    } else if (name == "KMEANS") {
	return Algorithm::KMEANS;
    }
    return Algorithm::UNKNOWN;
}

OutputStyle get_style_from_name(std::string name) {
    
    name = uppercase(name);

    if (name == "BORDER") {
	return OutputStyle::BORDER;
    } else if (name == "COLOR") {
	return OutputStyle::COLOR;
    }
    return OutputStyle::NONE;
}

struct AlgoParams {
    int threshold;
    int output_classes; // used if applicable
    OutputStyle style;
};


void help(std::ostream& out) {
    out << "Usage: " << std::endl
	<< "./gen -i <input> -o <output> -a <algorithm>" << std::endl;
}

void threshold(cv::Mat &input_mat, cv::Mat& output_mat, const AlgoParams& params) {
    cv::Mat thresh_mat;

    cv::threshold(input_mat, thresh_mat, params.threshold, 255, cv::THRESH_BINARY);

    cv::cvtColor(output_mat, thresh_mat, cv::COLOR_GRAY2BGR);
}

void threshold_otsu(cv::Mat& input_mat, cv::Mat& output_mat, const AlgoParams& params) {
    cv::Mat thresh_mat;
    
    cv::threshold(input_mat, thresh_mat, 0, 255, cv::THRESH_OTSU);

    cv::cvtColor(output_mat, thresh_mat, cv::COLOR_GRAY2BGR);
}

void kmeans_clustering(cv::Mat& input_mat, cv::Mat& output_mat, const AlgoParams& params) {
    
    cv::Mat best_labels;
    cv::Mat centers;
    

    
    int cols = input_mat.cols;
    int rows = input_mat.rows;
    
    cv::Mat pointsf;
    pointsf.create(cols * rows, 1, CV_32F);
    
    for (int row = 0; row < rows; row++) {
	const uint8_t* srcline = input_mat.ptr<uint8_t>(row);
	for (int col = 0; col < cols; col++) {

	    int off = row * cols + col;
	    
	    uint8_t pixel = srcline[col];
	    float val = static_cast<float>(pixel);
	    pointsf.at<float>(off, 0) = val;	    
	}
    }
    
    cv::TermCriteria criteria(cv::TermCriteria::Type::MAX_ITER, 100, 0.1f);
        
    int attempts = 10;
    int flags = cv::KMEANS_PP_CENTERS;
    
    cv::kmeans(pointsf, params.output_classes, best_labels, criteria, attempts, flags, centers);
    
    cv::Mat label_mat;
    label_mat.create(rows, cols, input_mat.type());

    
    label_mat = best_labels.reshape(1, {rows, cols});
        
    if (params.style == OutputStyle::COLOR) {
	color_regions(label_mat, output_mat, params.output_classes);
    } else {

	output_mat.create(rows, cols, input_mat.type());
	
	for (int row = 0; row < rows; row++) {
	    const int* srcline = label_mat.ptr<int>(row);
	    uint8_t* dstline = output_mat.ptr<uint8_t>(row);
	    for (int col = 0; col < cols; col++) {
		int label = srcline[col];
		float center = centers.at<float>(label);
		dstline[col] = static_cast<uint8_t>(center);
	    }
	}
    }
    
}

void watershed(cv::Mat& input_mat, cv::Mat& output_mat, const AlgoParams& params) {

    //cv::watershed(input_mat, markers_mat);
}

void labeling(cv::Mat& input_mat, cv::Mat& output_mat, const AlgoParams& params) {

    cv::Mat thresh_mat, label_mat, invert_mat;

    const int cols = input_mat.cols;
    const int rows = input_mat.rows;
    
    cv::threshold(input_mat, thresh_mat, params.threshold, 255, cv::THRESH_BINARY);
    //cv::threshold(input_mat, thresh_mat, 0, 255, cv::THRESH_OTSU);
    cv::bitwise_not(thresh_mat, invert_mat);
    
    int label_count = cv::connectedComponents(invert_mat, label_mat , 8, CV_32S, cv::CCL_SPAGHETTI);

    std::vector<uint64_t> sum_intensities;
    std::vector<int32_t> areas;
    std::vector<uint8_t> average_intensities;
    
    sum_intensities.resize(label_count);
    areas.resize(label_count);
    average_intensities.resize(label_count);
    
    for (int row = 0; row < rows; row++) {
	const uint8_t* __restrict__ input_line = input_mat.ptr<uint8_t>(row);
	const int32_t* __restrict__ labels_line = label_mat.ptr<int32_t>(row);
	
	for (int col = 0; col < cols; col++) {
	    int32_t label = labels_line[col];
	    uint8_t pix = input_line[col];

	    sum_intensities[label] += pix;
	    areas[label]++;
	}
    }

    for (int label = 0; label < label_count; label++) {
	average_intensities[label] = (sum_intensities[label] / areas[label]);
    }
    
    output_mat.create(rows, cols, CV_8UC3);

    if (params.style == OutputStyle::COLOR) {
	color_regions(label_mat, output_mat, label_count);
    } else {
	for (int row = 0; row < rows; row++) {
	    const int32_t* __restrict__ labels_line = label_mat.ptr<int32_t>(row);
	    cv::Vec3b* __restrict__ output_line = output_mat.ptr<cv::Vec3b>(row);
	    for (int col = 0; col < cols; col++) {
		int32_t label = labels_line[col];

		uint8_t avg = average_intensities[label];
		output_line[col] = cv::Vec3b(avg, avg, avg);
	    }
	}
    }    
}

constexpr uint32_t MAKE_COLOR(uint8_t r, uint8_t g, uint8_t b) {
    return (r << 24) | (g << 16) | (b << 8);    
}

void line_write_bgrcolor(uint8_t* __restrict__ line, int col, uint8_t r, uint8_t g, uint8_t b) {
    line[3 * col    ] = b;
    line[3 * col + 1] = g;
    line[3 * col + 2] = r;
}



void highlight_borders(cv::Mat& input_mat, cv::Mat& output_mat) {
    const int cols = input_mat.cols;
    const int rows = input_mat.rows;

    output_mat.create(rows * 2, cols * 2, CV_8UC3);    

    std::cout << "output: rows = " << output_mat.rows << ", cols = " << output_mat.cols << "\n";
    
    const cv::Vec3b COLOR_RED(0, 0, 255);
    
    for (int row = 0; row < rows - 1; row++) {
	const cv::Vec3b* __restrict__ input0 = input_mat.ptr<cv::Vec3b>(row);
	const cv::Vec3b* __restrict__ input1 = input_mat.ptr<cv::Vec3b>(row + 1);

	cv::Vec3b* __restrict__ output0 = output_mat.ptr<cv::Vec3b>(2 * row);	
	cv::Vec3b* __restrict__ output1 = output_mat.ptr<cv::Vec3b>(2 * row + 1);
	
	for (int scol = 0; scol < cols - 1; scol++) {
	    int dcol = scol * 2;
	    
	    cv::Vec3b pix = input0[scol];
	    cv::Vec3b pix1 = input0[scol + 1];
	    cv::Vec3b pix2 = input1[scol];
	    cv::Vec3b pix3 = input1[scol + 1];	    
	    
	    output0[dcol] = pix;
	    output0[dcol + 1] = pix;
	    output1[dcol] = pix;
	    output1[dcol + 1] = pix;	    
	    
	    if (pix != pix1) {
		output0[dcol + 1] = COLOR_RED;
	    }
	    if (pix != pix2) {
		output1[dcol] = COLOR_RED;
	    }
	    if (pix != pix3) {
		output1[dcol + 1] = COLOR_RED;
	    }

	}
    }
}

cv::Vec3b hsv_to_bgr(float h, float s, float v) {
    float c = v * s;
    float hp = h/60;
    float x = c * (1 - std::fabs(std::fmod(hp, 2) - 1));

    float r1, g1, b1;


    if (0 <= hp && hp < 1) {
	r1 = c; g1 = x; b1 = 0;
    } else if (1 <= hp && hp < 2) {
	r1 = x; g1 = c; b1 = 0;	
    } else if (2 <= hp && hp < 3) {
	r1 = 0; g1 = c; b1 = x;		
    } else if (3 <= hp && hp < 4) {
	r1 = 0; g1 = x; b1 = c;		
    } else if (4 <= hp && hp < 5) {
	r1 = x; g1 = 0; b1 = c;		
    } else {  // 5 <= hp && hp < 6
	r1 = c; g1 = 0; b1 = x;		
    } 

    float m = v - c;
    float rf, gf, bf;
    rf = r1 + m;
    gf = g1 + m;
    bf = b1 + m;

    cv::Vec3b rgb;
    rgb[2] = std::round(rf * 256);
    rgb[1] = std::round(gf * 256);
    rgb[0] = std::round(bf * 256);

    return rgb;
}

void initialize_palette(std::vector<cv::Vec3b>& palette, long seed) {

    std::mt19937 mt(seed);
    float h, s, v;

    std::uniform_real_distribution<> sat_distr(0.2, 1.0);
    std::uniform_real_distribution<> hue_distr(0, 360);
    std::uniform_real_distribution<> val_distr(0.2, 1.0);
    
    for (auto& elem: palette) {

	h = hue_distr(mt);
	s = sat_distr(mt);
	v = val_distr(mt);

	elem = hsv_to_bgr(h, s, v);
    }
    
    palette[0] = cv::Vec3b(0, 0, 0); // black background
    
}

void color_regions(cv::Mat& input_mat, cv::Mat& output_mat, int label_count) {

    std::vector<cv::Vec3b> palette;
    palette.resize(label_count);

    initialize_palette(palette, 0);
    
    const int rows = input_mat.rows;
    const int cols = input_mat.cols;    

    output_mat.create(rows, cols, CV_8UC3);
    
    for (int row = 0; row < rows; row++) {

	const int32_t* __restrict__ srcline = input_mat.ptr<int32_t>(row);
	uint8_t* __restrict__ dstline = output_mat.ptr<uint8_t>(row);
	for (int col = 0; col < cols; col++) {
	    int32_t label = srcline[col];

	    cv::Vec3b color = palette[label];

	    line_write_bgrcolor(dstline, col, color[0], color[1], color[2]);
	}
    }
    
}


void process(const std::string& input_filename, const std::string& output_filename,
	     Algorithm algorithm, const AlgoParams& params) {

    cv::Mat input_mat;
    cv::Mat output_mat;
    
    input_mat = cv::imread(input_filename, cv::IMREAD_GRAYSCALE);

    
    if (input_mat.empty()) {
	return;
    }

    output_mat.create(input_mat.rows, input_mat.cols, input_mat.type());
    
    
    switch (algorithm) {
    case Algorithm::THRESHOLD:	
	threshold(input_mat, output_mat, params);
	break;
    case Algorithm::OTSU:
	threshold_otsu(input_mat, output_mat, params);
	break;
    case Algorithm::CCL:
	labeling(input_mat, output_mat, params);
	break;
    case Algorithm::WATERSHED:
	watershed(input_mat, output_mat);
	break;
    case Algorithm::KMEANS:
	kmeans_clustering(input_mat, output_mat, params);
	break;
    default:
	break;
    }

    cv::Mat final_image = output_mat;
    switch (params.style) {
    case OutputStyle::BORDER:
	highlight_borders(output_mat, final_image);
	break;
    default:
	break;
    }
    
    cv::imwrite(output_filename, final_image);    
}

int main(int argc, char** argv) {

    struct optparse_long longopts[] = {
	{"help", 'h', OPTPARSE_NONE},
	{"input", 'i', OPTPARSE_REQUIRED},
	{"output", 'o', OPTPARSE_REQUIRED},
	{"alg", 'a', OPTPARSE_REQUIRED},
	{"thresh", 't', OPTPARSE_REQUIRED},
	{"style", 's', OPTPARSE_REQUIRED},
	{"classes", 'c', OPTPARSE_REQUIRED},
    };

    int opt, longindex;

    std::string input_filename, output_filename;
    Algorithm algorithm;
    AlgoParams params;

    // Initialize default params
    params.threshold = 127;
    params.output_classes = 8;
    params.style = OutputStyle::NONE;
    
    struct optparse options;
    optparse_init(&options, argv);

    while (options.optind < argc) {

	opt = optparse_long(&options, longopts, NULL);

	if (opt != -1) {

	    switch (opt) {
	    case 'h':
		break;
	    case 'i':
		input_filename = options.optarg;
		break;		
	    case 'o':
		output_filename = options.optarg;
		break;
	    case 'a':
		algorithm = get_algorithm_from_name(options.optarg);
		break;
	    case 't':
		params.threshold = std::stoi(options.optarg);
		break;
	    case 's':
		params.style = get_style_from_name(options.optarg);
		break;
	    case 'c':
		params.output_classes = std::stoi(options.optarg);
		break;
	    default:
		std::cerr << "Unrecognized options" << std::endl;

		
		help(std::cout);
		return -1;
	    }
	} else {
	    std::cerr << "Unrecognized option" << std::endl;

	    help(std::cout);
	    return -1;
	}
	
    }

    if (input_filename.empty()) {
	std::cerr << "Missing option: --input" << std::endl;
	help(std::cout);
	
	return -1;
    }
    
    process(input_filename, output_filename, algorithm, params);
    
    return 0;
}

