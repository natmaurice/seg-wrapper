#include <algorithm>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#define OPTPARSE_IMPLEMENTATION
#include <optparse.h>



enum class Algorithm {
    THRESHOLD,
    OTSU,
    CCL,
    WATERSHED,
    UNKNOWN,
};


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
    } 
    return Algorithm::UNKNOWN;
}


struct AlgoParams {
    int threshold;
};


void help(std::ostream& out) {
    out << "Usage: " << std::endl
	<< "./gen -i <input> -o <output> -a <algorithm>" << std::endl;
}

void threshold(cv::Mat &input_mat, cv::Mat output_mat, const AlgoParams& params) {
    cv::threshold(input_mat, output_mat, params.threshold, 255, cv::THRESH_BINARY);
}

void threshold_otsu(cv::Mat& input_mat, cv::Mat& output_mat, const AlgoParams& params) {
    cv::threshold(input_mat, output_mat, 0, 255, cv::THRESH_OTSU);
}

void labeling(cv::Mat& input_mat, cv::Mat& output_mat, const AlgoParams& params) {

    cv::Mat thresh_mat, label_mat;
    
    cv::threshold(input_mat, thresh_mat, 127, 255, cv::THRESH_BINARY);
    
    cv::connectedComponents(thresh_mat, label_mat , 8, CV_32S, cv::CCL_SPAGHETTI);

    
}

void watershed(cv::Mat& input_mat, cv::Mat& output_mat, const AlgoParams& params) {

    
}

void process(const std::string& input_filename, const std::string& output_filename,
	     Algorithm algorithm, const AlgoParams& params) {

    cv::Mat input_mat;
    cv::Mat output_mat;
    
    input_mat = cv::imread(input_filename);

    
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
    default:
	break;
    }

    cv::imwrite(output_filename, output_mat);
    
}

int main(int argc, char** argv) {

    struct optparse_long longopts[] = {
	{"help", 'h', OPTPARSE_NONE},
	{"input", 'i', OPTPARSE_REQUIRED},
	{"output", 'o', OPTPARSE_REQUIRED},
	{"alg", 'a', OPTPARSE_REQUIRED},
	{"thresh", 't', OPTPARSE_REQUIRED}	
    };

    int opt, longindex;

    std::string input_filename, output_filename;
    Algorithm algorithm;
    AlgoParams params;
    
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

    process(input_filename, output_filename, algorithm, params);
    
    return 0;
}

