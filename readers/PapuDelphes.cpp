#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <array>
#include <stdlib.h>
#include <functional>
#include <time.h>
#include <math.h>
#include <numeric>

#include "TROOT.h"

#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TLorentzVector.h"

#include "ExRootAnalysis/ExRootProgressBar.h"
#include "ExRootAnalysis/ExRootTreeBranch.h"
#include "ExRootAnalysis/ExRootTreeWriter.h"

using namespace std;

//---------------------------------------------------------------------------


static int NMAX = 7000;

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


template<int K>
class KMeans { 
public:
    KMeans(vector<PFCand*> particles, int max_iter=20) 
    {
        // randomly initialize centroids
        array<int, K> i_centroids;
        for (int i=0; i!=K; ++i) {
            while (true) {
                int i_p = rand() % particles.size();
                bool found = false;
                for (int j=0; j!=i; ++j) {
                    found = (i_centroids[j] == i_p);
                    if (found)
                        break;
                }
                if (!found) {
                    i_centroids[i] = i_p;
                    centroids[i][0] = particles[i_p]->eta;
                    centroids[i][1] = particles[i_p]->phi;
                    break;
                }
            }
        } 

        for (int i_iter=0; i_iter!=max_iter; ++i_iter) {
            assign_particles(particles);
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
            float eta = p->eta; float phi = p->phi;

            for (int i=0; i!=K; ++i) {
                auto dr = pow(eta - centroids[i][0], 2) + pow(phi - centroids[i][1], 2);
                if (dr < closest) {
                    closest = dr;
                    i_closest = i;
                }
            }
            clusters[i_closest].push_back(p);
        }
    }

    void update_centroids() 
    {
        for (int i=0; i!=K; ++i) {
            float eta_sum=0, phi_sum=0;
            auto &cluster = clusters[i];
            for (auto& p : cluster) {
                eta_sum += p->eta;
                phi_sum += p->phi;
            }
            centroids[i][0] = eta_sum / cluster.size();
            centroids[i][1] = phi_sum / cluster.size();
        }
    }
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
    _recursive_fit(const vector<PFCand*> &particles)
    {
        vector<vector<PFCand*>> clusters;

        auto kmeans = KMeans<K>(particles);
        for (int i_k=0; i_k!=K; ++i_k) {
            auto cluster = kmeans.get_clusters()[i_k];
            if (cluster.size() > N) {
                auto split_clusters = _recursive_fit(cluster);
                for (auto& c : split_clusters) {
                    clusters.push_back(c);
                }
            } else {
                clusters.push_back(cluster);
            } 
        }
        return clusters;
    }

};


//---------------------------------------------------------------------------

int main(int argc, char *argv[])
{

  srand(time(NULL));

  if(argc < 3) {
    cout << " Usage: " << "PapuDelphes" << " input_file"
         << " output_file" << endl;
    cout << " input_file - input file in ROOT format," << endl;
    cout << " output_file - output file in ROOT format" << endl;
    return 1;
  }

  // figure out how to read the file here 
  //

  TFile* ifile = TFile::Open(argv[1], "READ");
  TTree* itree = (TTree*)ifile->Get("Delphes;1");

  auto* fout = TFile::Open(argv[2], "RECREATE");
  auto* tout = new TTree("events", "events");

  unsigned int nevt = itree->GetEntries();
  TBranch* pfbranch = (TBranch*)itree->GetBranch("ParticleFlowCandidate");
  std::cout << "NEVT: " << nevt << std::endl;
  vector<PFCand> input_particles;

  vector<PFCand> output_particles;
  output_particles.reserve(7000);
  tout->Branch("particles", &output_particles);

  auto ho = HierarchicalOrdering<4, 10>();

  ExRootProgressBar progressBar(nevt);
  
  // add the pT of two PFCands
  auto add_pt = [](auto init, auto* b) { return init + b->pt; };
  // add the pT of a cluster of PFCands
  auto sum_pt = [&add_pt](auto &v) { return accumulate(v.begin(), v.end(), 0., add_pt); };
  // compare two clusters by sum pT
  auto comp_pt = [&sum_pt](auto &a, auto &b) { return sum_pt(a) > sum_pt(b); };

  for (unsigned int k=0; k<nevt; k++){
    itree->GetEntry(k);
    input_particles.clear();
    unsigned int npfs = pfbranch->GetEntries();
    npfs = itree->GetLeaf("ParticleFlowCandidate_size")->GetValue(0);
    //cout << "NPFS: " << npfs << endl;
    for (unsigned int j=0; j<npfs; j++){
      PFCand tmppf;
      tmppf.pt = itree->GetLeaf("ParticleFlowCandidate.PT")->GetValue(j);
      tmppf.eta = itree->GetLeaf("ParticleFlowCandidate.Eta")->GetValue(j);
      tmppf.phi = itree->GetLeaf("ParticleFlowCandidate.Phi")->GetValue(j);
      tmppf.e = itree->GetLeaf("ParticleFlowCandidate.E")->GetValue(j);
      tmppf.puppi = itree->GetLeaf("ParticleFlowCandidate.PuppiW")->GetValue(j);
      tmppf.hard_frac = itree->GetLeaf("ParticleFlowCandidate.hardfrac")->GetValue(j);
      tmppf.pdgid = itree->GetLeaf("ParticleFlowCandidate.PID")->GetValue(j);
      input_particles.push_back(tmppf);
      //cout << "PT: " << tmppf->pdgid << endl;
    }

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
    output_particles.resize(NMAX);

    tout->Fill();

    progressBar.Update(k, k);

  }

  fout->Write();
  fout->Close();

}
