<?php  require_once '/srv/www/mine/voteprism/inc.head.php'; ?>
   <main>
      <section>
         <pre>
By default, all accounts not pulled from a public government site are set private.

The only personal information, besides email and account votes, that we store is what you supply on the rankings page, and we only share any of it if you set your profile public (assumably to run for public office.)

We will use reasonable means to protect your email address, the only identification we require to hold your account votes.  If you do choose to give us your phone number and/or name we will share those details (along with registered email and votes) only while your account is set public and includes a link to your candidacy page (hosted elsewhere).  Even if the salting algorithm on our blockchain is somehow breached, updating it will be simple and can currently only reveal a UUID made only for our hash, linking your account to your vote only inside our database, ensuring access would still be needed to the physical database and only then could connect it to an email - we've literally spent years discussing how to protect your vote.

If you know how to check, you will plainly see the only javascript (a security risk) we reference only allows you to sort the data and is stored locally so even it can't be used as an attack.  We will strive to keep all your votes secure and confidential - the less personal information you give us the better (only an email is required).  If you choose the right motions, few politicians will rank high with you with only a few votes. 

If you wish to donate to the cause of increasing security, processing power or bandwidth, please click this link. (coming soon)

Hopefully by 2025 your donations will sponsor a hackathon designed to secure a globally available, open source version.
         </pre>
      </section>
   </main>
<?php  require_once '/srv/www/mine/voteprism/inc.foot.php'; ?>