#include <memory>
#include <print>

#include "bejzak_engine/common/util/engine_exception.h"
#include "bejzak_engine/vulkan/wrapper/util/check.h"
#include "bejzak_engine/common/file/standard_file_loader.h"
#include "bejzak_engine/vulkan/graphics_context/presentation.h"
#include "bejzak_engine/common/window/window_glfw.h"

int main() {
  try {
	auto fileLoader = std::make_unique<StandardFileLoader>();
	auto application = vlkn::Presentation::create(std::make_unique<WindowGlfw>("BejzakEngine", 1920, 1080), *fileLoader);
	application->run();
  }
  catch (const VkException& vkException) {
	  std::println("Vulkan exception occured with message: {} and VkResult code: {}", vkException.what(), static_cast<std::int32_t>(vkException.getResult()));
  }
  catch (const EngineException& engineException) {
	  std::println("Vulkan exception occured with message: {}. \n", engineException.what());
  }

  return EXIT_SUCCESS;
}