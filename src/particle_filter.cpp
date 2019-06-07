/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  // Set the number of particles.
  num_particles = 100;  

  // Add random Gaussian noise to each particle.
  std::default_random_engine gen;

  // Sensor noise distributions and associated variables.
  std::normal_distribution<double> x_dist(0, std[0]);
  std::normal_distribution<double> y_dist(0, std[1]);
  std::normal_distribution<double> theta_dist(0, std[2]);
  double x_noise = 0.0, y_noise = 0.0, theta_noise = 0.0;

  // Initializing all particles.
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    x_noise = x_dist(gen);
    y_noise = y_dist(gen);
    theta_noise = theta_dist(gen);

    p.id = i;
    p.x = x + x_noise;
    p.y = y + y_noise;
    p.theta = theta + theta_noise;

    // All initialized weights are set to 1
    p.weight = 1.0;

    particles.push_back(p);
  }
  // Set flage
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine gen;
  std::normal_distribution<double> x_dist(0, std_pos[0]);
  std::normal_distribution<double> y_dist(0, std_pos[1]);
  std::normal_distribution<double> theta_dist(0, std_pos[2]);
  double x_noise, y_noise, theta_noise = 0;

  for (int i = 0; i < num_particles; i++) {
    x_noise = x_dist(gen);
    y_noise = y_dist(gen);
    theta_noise = theta_dist(gen);

    // Case: car drives straight(yaw_rate<0.001)
    if (fabs(yaw_rate) < 0.001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta) + x_noise;
      particles[i].y += velocity * delta_t * sin(particles[i].theta) + y_noise;
      particles[i].theta += theta_noise;
    }
    else {
    // car is turning
      particles[i].x += velocity / yaw_rate
          * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) + x_noise;
      particles[i].y += velocity / yaw_rate
          * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t)) + y_noise;
      particles[i].theta += yaw_rate * delta_t + theta_noise;
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int i = 0; i < observations.size(); i++) {

      LandmarkObs curr_observation = observations[i];

      double min_distance = std::numeric_limits<double>::max();
      int match_index = std::numeric_limits<int>::min();;

      for (unsigned int j = 0; j < predicted.size(); j++) {
        LandmarkObs curr_landmark = predicted[j];

        double curr_distance = dist(curr_observation.x, curr_observation.y,
                                    curr_landmark.x, curr_landmark.y);

        if (curr_distance < min_distance) {
          min_distance = curr_distance;
          match_index = curr_landmark.id;
        }
      }

      observations[i].id = match_index;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i = 0; i < num_particles; i++) {
    Particle p = particles[i];
    double multivar_prod = 1;
    vector<LandmarkObs> landmarks_p;

    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      Map::single_landmark_s l = map_landmarks.landmark_list[j];
      if (dist(l.x_f, l.y_f, p.x, p.y) <= sensor_range){
        landmarks_p.push_back(LandmarkObs{l.id_i, l.x_f, l.y_f});
      }
    }

    vector<LandmarkObs> transformed_os;
    for (unsigned int j = 0; j < observations.size(); j++) {
      LandmarkObs o = observations[j], t;

      t.id = o.id;
      t.x = cos(p.theta)*o.x - sin(p.theta)*o.y + p.x;
      t.y = sin(p.theta)*o.x + cos(p.theta)*o.y + p.y;
      transformed_os.push_back(t);
    }

    dataAssociation(landmarks_p, transformed_os);

    for (unsigned int j = 0; j < transformed_os.size(); j++) {
      LandmarkObs o = transformed_os[j];
      LandmarkObs m;
      m.id = o.id;

      for (unsigned int k = 0; k < landmarks_p.size(); k++) {
        LandmarkObs p = landmarks_p[k];
        if (p.id == m.id) {
          m.x = p.x;
          m.y = p.y;
        }
      }

      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double new_weight = ( 1/(2*M_PI*std_x*std_y)) * exp( -( pow(m.x-o.x,2)/(2*pow(std_x, 2))
                      + (pow(m.y-o.y,2)/(2*pow(std_y, 2))) ) );

      multivar_prod *= new_weight;
    }
    particles[i].weight = multivar_prod;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles;
  std::default_random_engine gen;
   vector<double> weights;
    for (int i = 0; i < num_particles; i++) {
      weights.push_back(particles[i].weight);
  }

  for (int i = 0; i < num_particles; ++i) {
      std::discrete_distribution<> discrete_dist(weights.begin(), weights.end());
      new_particles.push_back(particles[discrete_dist(gen)]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}