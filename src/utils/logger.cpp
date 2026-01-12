#include "logger.h"
#include <iomanip>

namespace jin {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

const char* Logger::level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

const char* Logger::level_to_color(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "\033[36m";  // Cyan
        case LogLevel::INFO:  return "\033[32m";  // Green
        case LogLevel::WARN:  return "\033[33m";  // Yellow
        case LogLevel::ERROR: return "\033[31m";  // Red
        default: return "\033[0m";
    }
}

void Logger::log(LogLevel level, const std::string& msg) {
    if (level < level_) return;
    
    std::cout << level_to_color(level) 
              << "[" << level_to_string(level) << "] "
              << "\033[0m" 
              << msg << std::endl;
}

void Logger::debug(const std::string& msg) {
    log(LogLevel::DEBUG, msg);
}

void Logger::info(const std::string& msg) {
    log(LogLevel::INFO, msg);
}

void Logger::warn(const std::string& msg) {
    log(LogLevel::WARN, msg);
}

void Logger::error(const std::string& msg) {
    log(LogLevel::ERROR, msg);
}

std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    return oss.str();
}

} // namespace jin
