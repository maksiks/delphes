#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <array>
#include <stdlib>
#include <functional>
#include <time>

#include "TROOT.h"

#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"

#include "ExRootAnalysis/ExRootProgressBar.h"
#include "ExRootAnalysis/ExRootTreeBranch.h"
#include "ExRootAnalysis/ExRootTreeWriter.h"

using namespace std;

//---------------------------------------------------------------------------

static bool interrupted = false;

void SignalHandler(int sig)
{
  interrupted = true;
}


struct PFCand
{
  float pt = 0;
  float eta = 0;
  float phi = 0;
  float e = 0;
  float puppi = 1;
  float pdgid = 0;
  float hard_frac = 1;  
  float cluster_idx = -1;
};


template <int K, int N>
class HierarchicalOrdering {
public:
    HierarchicalOrdering() { }

    vector<vector<PFCand*>> 
    fit(vector<PFCand> &particles)
    {
        vector<PFCand*> p_particles;
        for (auto& p : particles)
            p_particles.push_back(&p);

        return _recursive_fit(p_particles);
    }
    
private:
    vector<vector<PFCand*>> 
     _recursive_fit(vector<PFCand*> &particles)
    {
        vector<vector<PFCand*>> clusters;

        kmeans = KMeans<K>(particles);
        for (int i_k=0; i_k!=K; ++i_k) {
            auto& cluster = kmeans.get_clusters()[i_k];
            if (cluster.size() > N) {
                split_clusters = _recursive_fit(cluster);
                for (auto& c : split_clusters) {
                    clusters.append(c);
                }
            } else {
                clusters.append(cluster);
            } 
        }
        return cluster;
    }

};


template<int K>
class KMeans { 
public:
    KMeans(vector<PFCand*> particles, int max_iter=20) 
    {
        // randomly initialize centroids
        array<int, K> i_centroids;
        for (int i=0; i!=K; ++i) {
            while (true) {
                auto i_p = rand() % particles.size();
                bool found = false;
                for (int j=0; j!=i; ++j) {
                    found = (i_centroids[j] == i_p);
                    if (found)
                        break;
                }
                if (!found) {
                    i_centroids[i] = i_p;
                    centroids[i][0] = particles[i_p]->Eta();
                    centroids[i][1] = particles[i_p]->Phi();
                    break;
                }
            }
        } 

        for (int i_iter=0; i_iter!=max_iter; ++i_iter) {
            assign_clusters(particles);
            update_centroids();
        }
    }

    ~KMeans() { }

    const array<vector<PFCand*>, K> get_clusters() { return clusters; }

private:
    array<array<float, 2>, K> centroids;
    array<vector<PFCand*>, K> clusters;

    void assign_particles(vector<PFCand*> &particles) 
    {
        for (int i=0; i!=K; ++i) {
            clusters[i].clear();
        }

        for (auto& p : particles) {
            float closest = 99999;
            int i_closest = -1;
            float eta = p->Eta(); float phi = p->Phi();

            for (int i=0; i!=K; ++i) {
                auto dr = (eta - clusters[i][0]) ** 2 + (phi - clusters[i][1]) ** 2;
                if (dr < closest) {
                    closest = dr;
                    i_closest = i;
                }
            }
            clusters[i_closest].push_back(&p);
        }
    }

    void update_centroids() 
    {
        for (int i=0; i!=K; ++i) {
            float eta_sum=0, phi_sum=0;
            auto &cluster = clusters[i];
            for (auto& p : cluster) {
                eta_sum += p->Eta();
                phi_sum += p->Phi();
            }
            centroids[i][0] = eta_sum / cluster.size();
            centroids[i][1] = phi_sum / cluster.size();
        }
    }
}; 


//---------------------------------------------------------------------------

int main(int argc, char *argv[])
{

  srand(time(NULL));

  if(argc < 3) {
    cout << " Usage: " << appName << " input_file"
         << " output_file" << endl;
    cout << " input_file - input file in ROOT format," << endl;
    cout << " output_file - output file in ROOT format" << endl;
    return 1;
  }

  // figure out how to read the file here 
  //

  auto* fout = TFile::Open(argv[2], "RECREATE");
  auto* tout = new TTree("events", "events");

  vector<PFCand> input_particles;
  vector<PFCand> output_particles;
  tout->Branch("particles", &output_particles);

  auto ho = HierarchicalOrdering<4, 10>();
  
  // add the pT of two PFCands
  auto add_pt = [](auto* a, auto* b) { return a->pt + b->pt; }
  // add the pT of a cluster of PFCands
  auto sum_pt = [](auto &v) { return accumulate(v.begin(), v.end(), 0, add_pt); }
  // compare two clusters by sum pT
  auto comp_pt = [](auto &a, auto &b) { return sum_pt(a) > sum_pt(b); }

  for (;;/*event loop*/) {
    input_particles.clear();
    // figure out how to push back the particles here
    
    // get clusters of 10 particles
    auto clusters = ho.fit(input_particles);
   
    // sort clusters by sum pT. not very efficient since we recompute
    // sum_pt for each comparison but whatever 
    sort(clusters.begin(), clusters.end(), comp_pt);

    output_particles.clear();
    int cluster_idx = 0;
    for (auto& cluster : clusters) {
      for (auto* p : cluster) {
        p->cluster_idx = cluster_idx;
        output_particles.push_back(*p); // *copy* the particle here before writing to tree
      }
      ++cluster_idx;
    }

    tout->Fill();
  }

  fout->Write();
  fout->Close();

}
