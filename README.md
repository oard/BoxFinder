# BoxFinder
Code and Data for the paper "Known by the Company it Keeps: Proximity-Based Indexing for Physical Content in Archival Repositories"

BoxFinder 1.0
Douglas W. Oard, oard@umd.edu
May 29, 2023

This project contains the following files:

titles.txt: The titles associated with each pdf file, which are used
to form title queries.  In this and other files, the first column is
the file name for the associated .pdf file (without the '.pdf' part).
In this file, the second column is the Brown University title metadata
for that PDF file.

foldermetadatshort.txt: The indexed content for short (hierarchical
subject-numeric code label expansion) metadata.

foldermedatalong.txt: The indexed content for long (short plus scope
note) metadata.

sncexpansion.xlsx: The data used to create short and long expansions of the 

folders.txt: The folder title for the folder containing each file,
based on Brown University metadata.  Used to generate sncexpansion.xlsx.

codes.txt: The subject-numeric code for the folder containing each
file, based on Brown Universty metadata.  Used to generate
sncexpansion.xlsx.  Note that some files have only code metadata,
others have only folder metadata, and many have both.

main.py: The Python code for boxfinder.  Note that your local
directory preix needs to be specified as a constant at the top of the
file.

You will also need the PDF files, which are available at https://users.umiacs.umd.edu/~oard/BoxFinder/pdf.zip: a compressed directory named pdf with a set of subdirectories for the PDF files, one subdrectory per box, named with the box number. Download and unzip this file.
