#pragma once

#include <string>
#include <sstream>
#include <iostream>

namespace jin {

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

class Logger {
public:
    static Logger& instance();
    
    void set_level(LogLevel level) { level_ = level; }
    LogLevel get_level() const { return level_; }
    
    void debug(const std::string& msg);
    void info(const std::string& msg);
    void warn(const std::string& msg);
    void error(const std::string& msg);
    
private:
    Logger() : level_(LogLevel::INFO) {}
    LogLevel level_;
    
    void log(LogLevel level, const std::string& msg);
    const char* level_to_string(LogLevel level);
    const char* level_to_color(LogLevel level);
};

inline void debug(const std::string& msg) { Logger::instance().debug(msg); }
inline void info(const std::string& msg) { Logger::instance().info(msg); }
inline void warn(const std::string& msg) { Logger::instance().warn(msg); }
inline void error(const std::string& msg) { Logger::instance().error(msg); }

std::string format_bytes(size_t bytes);

} // namespace jin
