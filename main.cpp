#include "application.h"

#include <memory>
#include <print>

#include "bejzak_engine/common/util/engine_exception.h"
#include "bejzak_engine/vulkan/wrapper/util/check.h"
#include "bejzak_engine/common/file/standard_file_loader.h"

int main() {
  try {
	Application app(std::make_unique<StandardFileLoader>());
	app.run();
  }
  catch (const VkException& vkException) {
	  std::println("Vulkan exception occured with message: {} and VkResult code: {}.", vkException.what(), static_cast<std::int32_t>(vkException.getResult()));
  }
  catch (const EngineException& engineException) {
	  std::println("Vulkan exception occured with message: {}.", engineException.what());
  }

  return EXIT_SUCCESS;
}