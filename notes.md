# Notes on ongoing work

## Notes 

- Embeddings at agent _and_ neighbourhood level (Capture differences in the space)

- Look at how neighbourhood type affects decision to move, and how neighbourhoods change as people move/leavel (feedback)

- Do it reach equilibruium?

- Utility?  -> afftect spread of opinions -> in homogeneous areas there are fewer opinions, more uniformity, less spread, in poorer areas that more people have access to there is more opinion

- Much richer model with more interesting (?) consequences

- Polaristation -> different parts of the embeddings might start to cluster
   -> spread of opinion dynamics

-  Richer descriptions to move on some of the simple canonical schelling / opinion dynamics models

[ ] Take an existing housing / gentrification / land-use model and enrich with embedings

-> Buxton becomes a reform area

[-] Can a decoder estimate the voting intention of someone based on their embeddings?

[ ] Look for some text data that I can use to build realistic embeddings

[ ] Ask chatgpt if there are some big surveys that I can use


## Plan 

 - [ ] Give all agents wealth (uniformly distributed 0-100) (later can relate to text embeddings)

 - [ ] Create neighbourhoods and calculate mean neighbourhood embedding from those of residents

 - [ ] Agent move decision now based on mean embedding of neighbourhood

 -> See what happens; expect similar results really


 -  [ ] Include wealth constraint in the move decision (assign each neighbourhood a 'cost' and restrict moves to agents with wealth > cost)

 -> Compare to pre-move. Do we start to see clustering on some embeddings and/or greater diversity of embeddings in some neighbourhoods?

