// Transport adapter only; geometry, descriptors and classifier are author GPD.
#include <gpd/grasp_detector.h>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  std::ifstream input(argv[2]);
  size_t n, k, m;
  if (!(input >> n >> k >> m) || !n || !k || !m) return 3;
  Eigen::Matrix3Xd cameras(3,k), samples(3,m);
  for (size_t i=0;i<k;i++) for(int j=0;j<3;j++) input >> cameras(j,i);
  gpd::util::PointCloudRGB::Ptr points(new gpd::util::PointCloudRGB);
  points->resize(n);
  Eigen::MatrixXi visibility(k,n);
  for (size_t i=0;i<n;i++) {
    input >> points->at(i).x >> points->at(i).y >> points->at(i).z;
    points->at(i).rgba=0xffffffff;
    for (size_t j=0;j<k;j++) input >> visibility(j,i);
  }
  for(size_t i=0;i<m;i++) for(int j=0;j<3;j++) input >> samples(j,i);
  if(!input) throw std::runtime_error("Truncated bridge input");
  std::srand(20261011);
  gpd::util::Cloud cloud(points,visibility,cameras);
  cloud.setSamples(samples);
  gpd::GraspDetector detector(argv[1]);
  detector.preprocessPointCloud(cloud);
  const auto hands=detector.detectGrasps(cloud);
  std::ofstream output(argv[3]);
  output << std::setprecision(17);
  for(const auto& hand:hands) {
    const auto& p=hand->getPosition();
    const auto& R=hand->getFrame();
    for(int i=0;i<3;i++) { for(int j=0;j<3;j++) output << R(i,j) << ' '; output << p(i) << ' '; }
    output << hand->getScore() << ' ' << hand->getGraspWidth() << ' '
           << hand->getBottom() << ' ' << hand->getTop() << '\n';
  }
  return 0;
}
