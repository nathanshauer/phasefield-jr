#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <fstream>

using namespace Eigen;

// Struct to represent a quadrature point
struct QuadraturePoint {
  double xi;      // xi coordinate
  double eta;     // eta coordinate
  double weight;  // weight associated with this point
};
std::vector<QuadraturePoint> create2x2QuadratureRule();
std::vector<QuadraturePoint> create3x3QuadratureRule();

// Adopting 2x2 quadrature rule
std::vector<QuadraturePoint> intrule = create2x2QuadratureRule();

struct Element {
  std::vector<int> node_ids;
};
struct Node {
  double x, y;
};
struct BC {
  int node;
  int type; // 0 dirichlet in x and y, 1 dirichlet in x, 2 dirichlet in y, 3 neumann
  double xval,yval;
};

// Solution vectors are global to be accessible at any point. This is not ideal, but since this is a simple example, it is acceptable.
VectorXd Uelas;
VectorXd Upf;

void createRectangularMesh(std::vector<Node> &nodes, std::vector<Element> &elements, int num_elements_x, int num_elements_y, double width, double height);
void assembleGlobalStiffness(MatrixXd &K, const std::vector<Element> &elements, const std::vector<Node> &nodes, double E, double nu, int nstate);
void computeElementStiffness(MatrixXd &Ke, const std::vector<Node> &nodes, const Element &element, double E, double nu);
void shapeFunctions(MatrixXd &N, MatrixXd &dN, const double qsi, const double eta);
void createB(MatrixXd &B, const MatrixXd &dN);
void applyBoundaryConditions(MatrixXd &K, VectorXd &F, const std::vector<BC> &bc_nodes);
void solveSystem(const MatrixXd &K, const VectorXd &F, VectorXd &U);
void generateVTKLegacyFile(const std::vector<Node> &nodes, const std::vector<Element> &elements, const std::string &filename);

int main() {
  // Define material properties
  double E = 30;    // Young's modulus in Pascals
  double nu = 0.2;  // Poisson's ratio

  // Define mesh
  int num_elements_x = 4;
  int num_elements_y = 4;
  double width = 4.;
  double height = 1.;
  double imposed_displacement_x = 0.2;

  // Data structures
  std::vector<Node> nodes;
  std::vector<Element> elements;
  std::vector<BC> bc_nodes;

  // Create regular rectangular mesh
  createRectangularMesh(nodes, elements, num_elements_x, num_elements_y, width, height);

  // Initialize global stiffness matrix and force vector
  int nstate_elas = 2, nstate_pf = 1;
  int ndofs_elas = nstate_elas * nodes.size(), ndofs_pf = nstate_pf * nodes.size();
  MatrixXd Kelas = MatrixXd::Zero(ndofs_elas, ndofs_elas);
  VectorXd Felas = VectorXd::Zero(ndofs_elas);
  Uelas = VectorXd::Zero(ndofs_elas);
  
  MatrixXd Kpf = MatrixXd::Zero(ndofs_pf, ndofs_pf);
  VectorXd Fpf = VectorXd::Zero(ndofs_pf);
  Upf = VectorXd::Zero(ndofs_pf);

  // Assemble global stiffness matrix
  assembleGlobalStiffness(Kelas, elements, nodes, E, nu, nstate_elas);

  // Print the global stiffness matrix
  // std::cout << "Global stiffness matrix K: \n" << K << std::endl;

  // Create boundary conditions  
  for (int i = 0; i < nodes.size(); ++i) {
    if (fabs(nodes[i].x) < 1.e-8) {
      bc_nodes.push_back({i, 0, 0.0, 0.0});     // Fix x displacement
    } else if (fabs(nodes[i].x - width) < 1.e-8) {
      bc_nodes.push_back({i, 1, imposed_displacement_x, 0.0});     // Impose total x displacemente on the right edge
    }
  }

  applyBoundaryConditions(Kelas, Felas, bc_nodes);

  // Print the global stiffness matrix after bc
  // std::cout << "Global stiffness matrix K_BC: \n" << K << std::endl;  

  // Solve the system
  solveSystem(Kelas, Felas, Uelas);

  generateVTKLegacyFile(nodes, elements, "output.vtk");

  // Output results
  // std::cout << "Displacements: \n" << U << std::endl;

  return 0;
}

void createRectangularMesh(std::vector<Node> &nodes, std::vector<Element> &elements, int num_elements_x, int num_elements_y, double width, double height) {
  // Generate nodes
  for (int j = 0; j <= num_elements_y; ++j) {
    for (int i = 0; i <= num_elements_x; ++i) {
      nodes.push_back({i * width / num_elements_x, j * height / num_elements_y});
    }
  }

  // Print the nodes
  std::cout << "Nodes:" << std::endl;
  for (const auto &node : nodes) {
    std::cout << "(" << node.x << ", " << node.y << ")" << std::endl;
  }  

  // Generate elements
  for (int j = 0; j < num_elements_y; ++j) {
    for (int i = 0; i < num_elements_x; ++i) {
      int n1 = j * (num_elements_x + 1) + i;
      int n2 = n1 + 1;
      int n3 = n1 + num_elements_x + 1;
      int n4 = n3 + 1;
      elements.push_back({{n1, n2, n4, n3}});
      // Print the coordinates of each node in the element
      std::cout << "Element nodes: ";
      for (int node_id : elements.back().node_ids) {
        std::cout << "(" << nodes[node_id].x << ", " << nodes[node_id].y << ") ";
      }
      std::cout << std::endl;
    }
  }
}

void assembleGlobalStiffness(MatrixXd &K, const std::vector<Element> &elements, const std::vector<Node> &nodes, double E, double nu, int nstate) {
  for (const auto &element : elements) {
    // Element stiffness matrix (for simplicity, assume a 4-node quadrilateral element)
    const int nquadnodes = 4;
    const int ndofel = nstate * nquadnodes;
    MatrixXd Ke = MatrixXd::Zero(ndofel, ndofel);
    computeElementStiffness(Ke, nodes, element, E, nu);

    // Assemble Ke into the global stiffness matrix K
    for (int i = 0; i < nquadnodes; ++i) {
      for (int j = 0; j < nquadnodes; ++j) {
        int row = nstate * element.node_ids[i];
        int col = nstate * element.node_ids[j];
        K(row, col) += Ke(nstate * i, nstate * j);
        K(row, col + 1) += Ke(nstate * i, nstate * j + 1);
        K(row + 1, col) += Ke(nstate * i + 1, nstate * j);
        K(row + 1, col + 1) += Ke(nstate * i + 1, nstate * j + 1);
      }
    }
  }
}

void computeElementStiffness(MatrixXd &Ke, const std::vector<Node> &nodes, const Element &element, double E, double nu) {
  // Extract node coordinates
  const Node &n1 = nodes[element.node_ids[0]];
  const Node &n2 = nodes[element.node_ids[1]];
  const Node &n3 = nodes[element.node_ids[2]];
  const Node &n4 = nodes[element.node_ids[3]];

  // Jacobian matrix
  double base = n2.x - n1.x;
  double height = n4.y - n1.y;
  double area = base * height;  // base * height since it is a simple mesh
  double detjac = area / 4.0;
  double dqsidx = 2.0 / base;    // for simple rectangular elements
  double dqsidy = 2.0 / height;  // for simple rectangular elements

  // Create diagonal matrix with dqsidx and dqsidy
  MatrixXd J_inv = MatrixXd::Zero(2, 2);
  J_inv(0, 0) = dqsidx;
  J_inv(1, 1) = dqsidy;

  // Plane stress elasticity matrix D
  double factor = E / (1 - nu * nu);
  MatrixXd D = MatrixXd::Zero(3, 3);
  D(0, 0) = factor;
  D(0, 1) = factor * nu;
  D(1, 0) = factor * nu;
  D(1, 1) = factor;
  D(2, 2) = factor * (1 - nu) / 2.0;

  // Compute B matrix
  MatrixXd B;
  for (const auto &qp : intrule) {
    MatrixXd N, dN;
    shapeFunctions(N, dN, qp.xi, qp.eta);

    // Transform derivatives to global coordinates
    MatrixXd dN_xy = J_inv.transpose() * dN.transpose();

    // Create B matrix
    createB(B, dN_xy.transpose());

    // Compute element stiffness matrix contribution of this integration point
    Ke += B.transpose() * D * B * qp.weight * detjac;
  }

  // Print the element stiffness matrix Ke
  // std::cout << "Element stiffness matrix Ke: \n"
  //           << Ke << std::endl;
}

void shapeFunctions(MatrixXd &N, MatrixXd &dN, const double qsi, const double eta) {
  double phi1qsi = (1 + qsi) / 2.0;
  double phi0eta = (1 - eta) / 2.0;
  double phi1eta = (1 + eta) / 2.0;
  double phi0qsi = (1 - qsi) / 2.0;

  Vector4d shape;
  shape << phi0qsi * phi0eta,
      phi1qsi * phi0eta,
      phi1qsi * phi1eta,
      phi0qsi * phi1eta;

  N = MatrixXd::Zero(2, 8);
  for (int i = 0; i < 4; ++i) {
    N(0, 2 * i) = shape(i);
    N(1, 2 * i + 1) = shape(i);
  }

  MatrixXd dNdqsi(4, 2);
  dNdqsi << 0.25 * (-1 + eta), 0.25 * (-1 + qsi),
      0.25 * (1 - eta), 0.25 * (-1 - qsi),
      0.25 * (1 + eta), 0.25 * (1 + qsi),
      0.25 * (-1 - eta), 0.25 * (1 - qsi);

  dN = dNdqsi;
}

void createB(MatrixXd &B, const MatrixXd &dN) {
  B = MatrixXd::Zero(3, 8);
  for (int i = 0; i < 4; ++i) {
    B(0, 2 * i) = dN(i, 0);
    B(1, 2 * i + 1) = dN(i, 1);
    B(2, 2 * i) = dN(i, 1);
    B(2, 2 * i + 1) = dN(i, 0);
  }
}

void applyBoundaryConditions(MatrixXd &K, VectorXd &F, const std::vector<BC> &bc_nodes) {
  for (const auto &bc : bc_nodes) {
    int row = 2 * bc.node;
    if (bc.type == 0) {
      // Dirichlet in x and y
      F -= K.col(row) * bc.xval;
      F -= K.col(row+1) * bc.yval;
      K.row(row).setZero();
      K.col(row).setZero();
      K.row(row+1).setZero();
      K.col(row+1).setZero();
      K(row, row) = 1.0;      
      K(row + 1, row + 1) = 1.0;
      F(row) = bc.xval;
      F(row + 1) = bc.yval;
    } else if (bc.type == 1) {
      // Dirichlet in x      
      F -= K.col(row) * bc.xval;
      K.row(row).setZero();
      K.col(row).setZero();      
      K(row, row) = 1.0;
      F(row) = bc.xval;
    } else if (bc.type == 2) {
      // Dirichlet in y
      F -= K.col(row+1) * bc.yval;
      K.row(row+1).setZero();
      K.col(row+1).setZero();      
      K(row + 1, row + 1) = 1.0;
      F(row + 1) = bc.yval;
    } else if (bc.type == 3) {
      // Neumann
      F(row) += bc.xval;
      F(row + 1) += bc.yval;
    }
  }
}

void solveSystem(const MatrixXd &K, const VectorXd &F, VectorXd &U) {
  U = K.fullPivLu().solve(F);
  std::cout << "Displacements: \n" << U << std::endl;
}

void generateVTKLegacyFile(const std::vector<Node> &nodes, const std::vector<Element> &elements, const std::string &filename) {
  std::ofstream vtkFile(filename);  

  vtkFile << "# vtk DataFile Version 2.0\n";
  vtkFile << "FEM results\n";
  vtkFile << "ASCII\n";
  vtkFile << "DATASET UNSTRUCTURED_GRID\n";

  // Write points
  vtkFile << "POINTS " << nodes.size() << " float\n";
  for (const auto &node : nodes) {
    vtkFile << node.x << " " << node.y << " 0.0\n";
  }

  // Write cells
  vtkFile << "CELLS " << elements.size() << " " << elements.size() * 5 << "\n";
  for (const auto &element : elements) {
    vtkFile << "4 " << element.node_ids[0] << " " << element.node_ids[1] << " " << element.node_ids[2] << " " << element.node_ids[3] << "\n";
  }

  // Write cell types
  vtkFile << "CELL_TYPES " << elements.size() << "\n";
  for (size_t i = 0; i < elements.size(); ++i) {
    vtkFile << "9\n";  // VTK_QUAD
  }

  // Write point data (displacements)
  vtkFile << "POINT_DATA " << nodes.size() << "\n";
  vtkFile << "VECTORS displacements float\n";
  for (size_t i = 0; i < nodes.size(); ++i) {
    vtkFile << Uelas(2 * i) << " " << Uelas(2 * i + 1) << " 0.0\n";
  }

  // Write point data (phase field)
  vtkFile << "SCALARS phasefield float 1\n";
  vtkFile << "LOOKUP_TABLE default\n";
  for (size_t i = 0; i < nodes.size(); ++i) {
    vtkFile << Upf(i) << "\n";
  }

  vtkFile.close();

}

// -------------------------- Quadrature rules --------------------------

std::vector<QuadraturePoint> create2x2QuadratureRule() {
  // 2-point Gaussian quadrature positions and weights
  const double points[2] = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
  const double weights[2] = {1.0, 1.0};

  std::vector<QuadraturePoint> rule;

  // Create 4 quadrature points (2x2 grid)
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      QuadraturePoint qp;
      qp.xi = points[i];
      qp.eta = points[j];
      qp.weight = weights[i] * weights[j];
      rule.push_back(qp);
    }
  }

  return rule;
}

std::vector<QuadraturePoint> create3x3QuadratureRule() {
  // 3-point Gaussian quadrature weights and positions
  const double points[3] = {-std::sqrt(3.0 / 5.0), 0.0, std::sqrt(3.0 / 5.0)};
  const double weights[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};

  std::vector<QuadraturePoint> rule;

  // Create 9 quadrature points
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      QuadraturePoint qp;
      qp.xi = points[i];
      qp.eta = points[j];
      qp.weight = weights[i] * weights[j];
      rule.push_back(qp);
    }
  }

  return rule;
}