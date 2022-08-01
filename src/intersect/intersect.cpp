#include "AllIntersectRenderer.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ext/stb_image_write.h"

/* namespace xd code */
namespace dx {
   
  extern "C" int main(int ac, char **av)
  {
    try {
      AllIntersectRenderer sample;
    
      int sx = 1200;
      int sy = 1024;
      sample.resize(sx, sy);
      sample.render();

      std::vector<uint32_t> pixels(sx*sy);
      sample.downloadPixels(pixels.data());

      const std::string fileName = "osc_example2.png";
      stbi_write_png(fileName.c_str(),sx,sy,4,
                     pixels.data(),sx*sizeof(uint32_t));
      std::cout << std::endl
                << "Image rendered, and saved to " << fileName << " ... done." << std::endl
                << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << "FATAL ERROR: " << e.what() << std::endl;
      exit(1);
    }
    return 0;
  }

} // :: dx