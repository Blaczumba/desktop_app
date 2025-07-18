#include "application_base.h"

#include "bejzak_engine/sources/window/window_glfw.h"

ApplicationBase::ApplicationBase() {
	_window = Window::createWindow("Bejzak Engine", 1920, 1080).value();
	std::vector<const char*>requiredExtensions = _window->getExtensions();
#ifdef VALIDATION_LAYERS_ENABLED
	requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif // VALIDATION_LAYERS_ENABLED

	_instance = Instance::create("Bejzak Engine", requiredExtensions).value();
#ifdef VALIDATION_LAYERS_ENABLED
	_debugMessenger = std::make_unique<DebugMessenger>(*_instance);
#endif // VALIDATION_LAYERS_ENABLED

	_surface = _window->createSurface(*_instance).value();
	_physicalDevice = PhysicalDevice::create(*_surface).value();
	_logicalDevice = LogicalDevice::create(*_physicalDevice).value();
	_programManager = std::make_unique<ShaderProgramManager>(*_logicalDevice);
	_swapchain = Swapchain::create(*_logicalDevice).value();

	_singleTimeCommandPool = std::make_unique<CommandPool>(*_logicalDevice);
}
